# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0.1, type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output, pos_out, src_pos, tgt_pos  = model(**sample['net_input'])
        pos_out_logits = pos_out
        src_pos_logits = src_pos[0]
        pos_token = model.get_srcpos(sample, net_output).view(-1, 1)
        non_pad_mask = pos_token.ne(self.padding_idx)
        kl_loss = self.kl_categorical(
            pos_out_logits.view(-1, pos_out_logits.size(-1)), src_pos_logits.view(-1, src_pos_logits.size(-1)),
            non_pad_mask, None)
        nmt_loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        crf_loss, _ = self.compute_loss(model, [pos_out], sample, reduce=reduce, pos="src")
        crf_loss3, _ = self.compute_loss(model, src_pos, sample, reduce=reduce, pos="src")
        crf_loss2, _ = self.compute_loss(model, tgt_pos, sample, reduce=reduce, pos="tar")

        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': nmt_loss.data,
            'nll_loss': nll_loss.data,
            'src_pos_loss':crf_loss.data,
            'tgt_pos_loss': crf_loss2.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        loss = nmt_loss + crf_loss + crf_loss2 + crf_loss3 + kl_loss
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True, pos=None):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        if pos == "src":
            target = model.get_srcpos(sample, net_output).view(-1, 1)
        elif pos == "tar":
            target = model.get_tarpos(sample, net_output).view(-1, 1)
        else:
            target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,)
        return loss, nll_loss

    def kl_categorical(self, p_logit, q_logit, non_pad_mask, gt_mask=None):
        non_pad_mask = non_pad_mask.view(-1)
        p = F.softmax(p_logit, dim=-1)
        _kl = torch.mean(p * (F.log_softmax(p_logit, dim=-1) - F.log_softmax(q_logit, dim=-1)), 1)[non_pad_mask.bool()]
        return torch.sum(_kl)

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        src_pos_loss_sum = sum(log.get('src_pos_loss', 0) for log in logging_outputs)
        tgt_pos_loss_sum = sum(log.get('tgt_pos_loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('src_pos_loss', src_pos_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('tgt_pos_loss', tgt_pos_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

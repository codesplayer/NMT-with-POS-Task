# Neural Machine Translation with Integrating Part of Speech Task

## Introduction

This is the code for the paper "Neural Machine Translation with Integrating Part of Speech Task" .  The implementation is on top of the open-source NMT toolkit [Fairseq](https://github.com/pytorch/fairseq). You might need to glance over the user manual of Fairseq for knowing the basic usage of Fairseq.

## Prerequisites

1. [PyTorch](http://pytorch.org/) version >= 1.4.0
2. Python version >= 3.6

## Train

For low resources translation tasks:

```
code_path=${path_code}
src=${source_language}
data=${data_path}
tgt=${target_language}

models=${save_models}

log=${Tensorboad_log_path}

CUDA_VISIBLE_DEVICES=6 nohup > ./train.log 2>&1 python $code_path/train.py --fp16  ${data}/save/ \
    --source-lang $src --target-lang $tgt \
    --arch transformer_wmt_en_de --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0007 --min-lr 1e-09 --max-epoch 40 \
    --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1\
    --max-tokens 4096  --update-freq 4 \
    --no-progress-bar   --tensorboard-logdir ${log} \
    --save-dir ${models}  &
```

For resource-rich translation tasks:

```
code_path=${path_code}
src=${source_language}
tgt=${target_language}
data=${data_path}
models=${save_models}

log=${Tensorboad_log_path}

CUDA_VISIBLE_DEVICES=6 nohup > ./en2de.log 2>&1 python $code_path/train.py --fp16  ${data_path}/save/ \
    --source-lang $src --target-lang $tgt --share-decoder-input-output-embed \
    --arch transformer_wmt_en_de --max-update 200000  \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-09 --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0007 --min-lr 1e-09  --weight-decay 0.0 --save-interval-updates 2000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1\
    --max-tokens 4096  --update-freq 4  --log-interval 500  \
    --no-progress-bar   --tensorboard-logdir ${log} \
    --save-dir ${models}  &
```

We use the parameters in low_data.sh to train low-resource translation tasks. And We use the parameters in large_data.sh to train Abundant resources translation tasks

## Inference

```
model=${model_path}
data=${data_path}
result=${result_log_path}
code_path=${path_code}
CUDA_VISIBLE_DEVICES=4 python ${path_code}/generate.py .${data_path}/save \
    --source-lang en --target-lang de \
    --path ${model_path} --batch-size 16 --lenpen 0.6 \
    --beam 24 --remove-bpe 2>&1 | tee ${result_log_path}/result.log
```


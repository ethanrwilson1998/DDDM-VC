#!/bin/bash
# Usage: sh infer.sh <src_path> <trg_path> <output_dir> <epsilon> <theta>
# Specify epsilon as 0 if wanting to use theta

ARG1=${1:-'<src_path>'}  # not optional
ARG2=${2:-'<trg_path>'}  # not optional
ARG3=${3:-'./converted/out.wav'}
ARG4=${4:-0}
ARG5=${5:-0}

python inference.py \
    --src_path $ARG1 \
    --trg_path $ARG2 \
    --ckpt_model './ckpt/model_base.pth' \
    --ckpt_voc './vocoder/voc_ckpt.pth' \
    --ckpt_f0_vqvae './f0_vqvae/f0_vqvae.pth' \
    --output_path $ARG3 \
    -t 100 \
    --epsilon $ARG4 \
    --theta $ARG5
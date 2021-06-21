#!/bin/bash

# Exit on error
set -e
set -o pipefail

# Main storage directory.
# If you start from downloading MUSDB18, you'll need disk space to dump the MUSDB18 and its wav.
musdb18_dir=

# After running the recipe a first time, you can run it from stage 1 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=python

# Example usage
# ./run.sh --stage 1 --tag my_tag

# General
stage=0  # Controls from which stage to start
tag=  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=$CUDA_VISIBLE_DEVICES

# Data
sample_rate=44100

# Training
batch_size=8
num_workers=4
#optimizer=adam
lr=0.001
epochs=

# Architecture
n_blocks=8
n_repeats=3
mask_nonlinear=relu

# Evaluation
eval_use_gpu=0

. utils/parse_options.sh

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${uuid}
fi
expdir=exp/train_convtasnet_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"

if [[ $stage -le 1 ]]; then
    echo "Stage 1: Training"
    mkdir -p logs
    CUDA_VISIBLE_DEVICES=$id $python_path train.py \
		--train_dir $musdb18_dir \
		--sample_rate $sample_rate \
		--lr $lr \
		--epochs $epochs \
		--batch_size $batch_size \
		--num_workers $num_workers \
		--mask_act $mask_nonlinear \
		--n_blocks $n_blocks \
		--n_repeats $n_repeats \
		--exp_dir ${expdir} | tee logs/train_${tag}.log
	cp logs/train_${tag}.log $expdir/train.log
fi

if [[ $stage -le 2 ]]; then
	echo "Stage 2 : Evaluation"
	CUDA_VISIBLE_DEVICES=$id $python_path eval.py \
		--use_gpu $eval_use_gpu \
        --root $musdb18_dir \
		--exp_dir ${expdir} | tee logs/eval_${tag}.log
	cp logs/eval_${tag}.log $expdir/eval.log
fi

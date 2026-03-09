#!/bin/bash
WANDB_PROJECT="selective-pruning"
DTYPE="fp32" # "bfp16" or "int8" if CPU or Low VRAM


# poetry run python prune.py "facebook/galactica-125m" --dtype $DTYPE \
#     --wandb_project $WANDB_PROJECT \
#     --focus "pile_codeless" --cripple "code" \
#     --token_limit 1000 \
#     --ff_frac 0.02 --attn_frac 0.00 \
#     --collection_sample_size 1e5 \
#     --eval_sample_size 1e5 \
#     --run_pre_test True \
#     --name "$facebook/galactica-125m $retain_set code"

for i in $(seq 1000000); do
        poetry run python prune.py "google/vit-base-patch16-224"\
                --dtype $DTYPE \
                --wandb_project $WANDB_PROJECT \
                --focus   "imagenet-1k-birdless" \
                --cripple "imagenet-1k-birds" \
                --token_limit 1000 \
                --ff_frac 0.02 --attn_frac 0.00 \
                --n_steps 1\
                --collection_sample_size 1e5 \
                --eval_sample_size 1e5 \
                --recalculate_activations false \
                --misc 1 \
                --name "google/vit-base-patch16-224 imagenet-1k-birdless imagenet-1k-birds"   # run them serially; or start several in parallel if you have multiple GPUs
done

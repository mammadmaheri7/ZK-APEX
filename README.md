#  ZK-APEX
This repo provides the code to recreate the results of ZK-APEX. The base implementation is adapted from [Taker](https://github.com/nickypro/taker).

## Repo Overview
The experiment folders, specifically `exp1-vit-pruning/` and `exp2-llm-pruning/`, can be used to duplicate the findings of the paper. The implementation of the pruning can be found in `taker_mmd/src/taker/weight_pruning.py` and `taker_mmd/src/taker/weight_prunin_llm`.

## Setup
The experiments are best run with Python 3.10. 

Run the following in the root directory.
``` 
$ pip install -r
```
and

```
$ pip install ./taker
```

Finally, access to the Imagenet-1k is required. Navigate, [here](https://huggingface.co/datasets/ILSVRC/imagenet-1k) and request access. Account creation may be required.

## Running experiments
Before running, run `export HF_TOKEN="TOKEN"`, with your hugging face access token. Run `run-llm-pruning.sh` or `run-vit-pruning.sh` to reproduce results. The hyperparameters can be changed in the bash files.



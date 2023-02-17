import torch
import numpy as np
import os
from parser import SAVE_DIR, parse_args


def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

def get_ckpt_steps(ckpt_files):
    steps = [int(file.split('_')[-1][:-8]) for file in ckpt_files]
    root = ckpt_files[0].split('_')[:-1]
    
    return steps, root

if __name__ == '__main__':
    args = parse_args()

    # Get checkpoints of experiment
    ckpt_files = recursive_glob(os.path.join(SAVE_DIR, args.expt_name))
    ckpt_steps, file_root = get_ckpt_steps(ckpt_files)

    # init
    n = len(ckpt_files)
    cosine_similiarities = np.zeros((n,n))
    prediction_disagreement = np.zeros((n,n))

    # Comparison all-to-all 
    # NOTE in the future can also add comparison to a single checkpoint (e.g. last)
    for i in range(n):
        model_i = ... # TODO
        for j in range(n):
            if i == j:
                cosine_similiarities[i,i] = 1
                prediction_disagreement[i,i] = 0
            else:

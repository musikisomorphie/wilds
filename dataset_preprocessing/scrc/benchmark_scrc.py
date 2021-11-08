import pandas as pd
import pathlib
import argparse
import torch
import random
import os
import sys
import numpy as np
import copy
from matplotlib import pyplot as plt
from tqdm import tqdm
from collections import defaultdict


def compute_stat(root_dir, method, data):
    if 'scrc' in data:
        phase = ['train', 'val', 'test']
        subtyp = ['CMS1', 'CMS2', 'CMS3', 'CMS4']
    elif 'rxrx1' in data:
        phase = ['train', 'val', 'test', 'id_test']
        subtyp = ['HEPG2', 'HUVEC', 'RPE', 'U2OS']
    else:
        raise ValueError('the {} data is not supported'.format(data))
    subtyp += ['total']

    exps = []
    for dir in root_dir.glob('*'):
        if method in dir.name and data in dir.name:
            exps.append(dir)

    res = dict.fromkeys(phase, None)
    for phs in phase:
        res[phs] = dict.fromkeys(subtyp, None)
        for exp in exps:    
            print(exp)
            df = pd.read_csv(exp / '{}_eval.csv'.format(phs))
            idx = np.argmax(df['acc_avg'])
            # this is necessary, avoid list copy by reference
            if res[phs]['total'] is None:
                res[phs]['total'] = list()
            res[phs]['total'].append(df['acc_avg'][idx])
            for sid, sub in enumerate(subtyp[:-1]):
                # print(sub)
                col = list(df.columns)[(sid + 1) * 2]
                if 'scrc' in data:
                    assert int(sub[-1]) - 1 == int(col[-1])
                elif 'rxrx1' in data:
                    assert sub == col.split(':')[-1]

                sub_acc = df[col][idx]
                if res[phs][sub] is None:
                    res[phs][sub] = list()
                res[phs][sub].append(sub_acc)
    print(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir',
                        type=pathlib.Path,
                        required=True)
    args = parser.parse_args()
    compute_stat(args.root_dir, 'ERM', 'scrc_012')

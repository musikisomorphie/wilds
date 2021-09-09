import pandas as pd
import pathlib
import argparse
import torch
import random
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from collections import defaultdict


def process_scrc(root_dir, val_pat=2, treg=3):
    root_dir = pathlib.Path(root_dir)
    imgs, labs = [], []
    tma_reg, lens = [], [0]
    for reg in range(treg):
        scrc_pt = root_dir / 'scrc_symm_circle_{}.pt'.format(reg)
        img, lab = torch.load(str(scrc_pt))
        # this axis shift makes left_pad easier
        # for normflow computation
        # because of downscale the rgb value is rather a float 254.242 than 254.0
        # lead to minor diff when call byte()
        img = img[:, [4, 0, 1, 2]].byte()
        # remove mucin (5) and misc (3) cell annotation
        # times the label with 10 for the division by 256
        _ncl = img[:, 0]
        _ncl_idx = (_ncl == 3) | (_ncl == 5)
        _ncl[_ncl_idx] = 0
        img[:, 0] = _ncl * 10
        print(torch.unique(img[:,0]))

        lab = lab.float()

        # save float tensor instead of double tensor
        imgs.append(img)
        labs.append(lab)
        lens.append(img.shape[0])
        tma_reg += [reg] * img.shape[0]

        # save float tensor for normalizing flow
        scrc_out_pt = str(root_dir / 'scrc_wilds_{}.pt'.format(reg))
        with open(scrc_out_pt, 'wb') as f:
            torch.save((img, lab), f)
            print('save {} done'.format(scrc_out_pt))

    imgs = torch.cat(imgs)
    labs = torch.cat(labs)
    lens = np.cumsum(lens)
    print(imgs.type(), labs.type(), imgs.shape, labs.shape, lens)

    imgs_path = str(root_dir / 'scrc_wilds_img.pt')
    with open(imgs_path, 'wb') as f:
        torch.save(imgs, f)
        print('save {} done'.format(imgs_path))

    labs_path = str(root_dir / 'scrc_wilds_lab.pt')
    with open(labs_path, 'wb') as f:
        torch.save(labs, f)
        print('save {} done'.format(labs_path))

    scrc_dict = {'tma_id': [],
                 'dataset': None,
                 'tma_reg': [],
                 'cms': [],
                 'pat_id': []}

    scrc_dict['tma_id'] = [i for i in range(imgs.shape[0])]
    scrc_dict['tma_reg'] = tma_reg
    cms = labs[:, -4:]
    cms = torch.argmax(cms, dim=1)
    scrc_dict['cms'] = cms.numpy().tolist()
    scrc_dict['pat_id'] = labs[:, 0].int().numpy().tolist()

    for reg in range(treg):
        scrc_dict['dataset'] = np.asarray(['train'] * imgs.shape[0])
        scrc_reg = labs[lens[reg]:lens[reg + 1], 0]

        val_msk = np.zeros(imgs.shape[0], dtype=np.bool)
        tst_msk = np.zeros(imgs.shape[0], dtype=np.bool)
        pat_id = np.unique(scrc_reg)
        # make sure that the validation tmas include
        # 2 spots / patient
        for pat in pat_id:
            tma_id = np.where(scrc_reg == pat)[0].tolist()
            tma_id = [tid + lens[reg] for tid in tma_id]
            random.shuffle(tma_id)
            print(tma_id)
            # keep 2 val spots for each patient
            # if possible
            cut_id = 0 if len(tma_id) < val_pat else val_pat
            val_msk[tma_id[:cut_id]] = True
            tst_msk[tma_id[cut_id:]] = True

        val_sum = val_msk[val_msk == True].sum()
        tst_sum = tst_msk[tst_msk == True].sum()
        assert val_sum + tst_sum == scrc_reg.shape[0]
        scrc_dict['dataset'][val_msk] = 'val'
        scrc_dict['dataset'][tst_msk] = 'test'
        scrc_dict['dataset'] = scrc_dict['dataset'].tolist()

        scrc_df = pd.DataFrame(scrc_dict)
        if reg == 0:
            csv_id = '120'
        elif reg == 1:
            csv_id = '201'
        elif reg == 2:
            csv_id = '012'

        scrc_df_path = pathlib.Path(
            root_dir) / 'scrc_wilds_{}.csv'.format(csv_id)
        scrc_df.to_csv(str(scrc_df_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir',
                        type=pathlib.Path,
                        required=True)
    args = parser.parse_args()
    process_scrc(args.root_dir)

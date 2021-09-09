import unittest
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import itertools
import json
import re
import csv
import os
import argparse
import torch
import roman
import copy
import time
import sys
sys.path.append('.')


class test_scrc(unittest.TestCase):
    def __init__(self, scrc_dir, val_pat=2):
        super().__init__()

        scrc_dir = pathlib.Path(scrc_dir)
        imgs_path = scrc_dir / 'scrc_wilds_img.pt'
        imgs = torch.load(str(imgs_path))
        imgs = imgs[:, [1, 2, 3, 0]]
        imgs[:, -1] = imgs[:, -1] / 10
        labs_path = scrc_dir / 'scrc_wilds_lab.pt'
        labs = torch.load(str(labs_path))
        lens = [0]
        for i in range(3):
            pt_path = scrc_dir / 'scrc_symm_circle_{}.pt'.format(i)
            img, lab = torch.load(str(pt_path))
            # because of downscale the rgb value is rather a float 254.242 than 254.0
            # lead to minor diff when call byte()
            img = img[:, [0, 1, 2, 4]].byte()
            ncls = img[:, -1]
            ncls[(ncls == 3) | (ncls == 5)] = 0
            img[:, -1] = ncls
            lab = lab.float()
            img_part = imgs[lens[-1]: lens[-1] + lab.shape[0]]
            lab_part = labs[lens[-1]: lens[-1] + lab.shape[0]]
            # print(img_part)
            self.assertTrue(torch.all(img_part == img))
            self.assertTrue(torch.all(lab_part == lab))
            lens.append(lens[-1] + lab.shape[0])

        pat_id = labs[:, 0].numpy()
        cms_vote = labs[:, -4:].numpy()
        df_dict = dict()
        for i in ('012', '120', '201'):
            df_path = scrc_dir / 'scrc_wilds_{}.csv'.format(i)
            df = pd.read_csv(str(df_path))

            for index, row in df.iterrows():
                self.assertTrue(pat_id[index] == row['pat_id'])
                self.assertTrue(
                    cms_vote[index, row['cms']] == cms_vote[index].max())

            trn = df.loc[df['dataset'] == 'train']
            self.assertEqual(set(np.unique(trn['tma_reg']).tolist()),
                             set([int(i[0]), int(i[1])]))

            val = df.loc[df['dataset'] == 'val']
            self.assertEqual(np.unique(val['tma_reg']).tolist(),
                             [int(i[2])])

            tst = df.loc[df['dataset'] == 'test']
            self.assertEqual(np.unique(tst['tma_reg']).tolist(),
                             [int(i[2])])

            print(i, np.unique(trn['tma_reg']).tolist(), len(val))

            val_p = pat_id[val['tma_id']]
            pat, cnt = np.unique(val_p, return_counts=True)
            self.assertTrue(all(cnt == val_pat))


def main(args):
    scrc = test_scrc(args.scrc_dir)
    print('unittest scrc done!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Test the data conversion for wilds repo')
    parser.add_argument('--scrc-dir',
                        type=pathlib.Path,
                        default='/home/histopath/Data/SCRC_nuclei/TMA_spots',
                        metavar='DIR',
                        help='path the Swiss CRC dataset')
    args = parser.parse_args()

    main(args)

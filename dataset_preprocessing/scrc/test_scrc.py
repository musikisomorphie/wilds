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
from PIL import Image
sys.path.append('.')


class test_scrc(unittest.TestCase):
    def __init__(self, old_dir, new_dir, val_pat=2):
        super().__init__()

        new_dir = pathlib.Path(new_dir)
        old_dir = pathlib.Path(old_dir)
        with open(str(new_dir / 'clinical_data_wilds.json'), 'r') as jfile:
            meta_new = json.load(jfile)
        with open(str(old_dir / 'clinical_data_cms.json'), 'r') as jfile:
            meta_old = json.load(jfile)

        for pat in meta_new:
            pat_new = meta_new[pat]
            pat_old = meta_old[pat]
            for pat_info in pat_new:
                if 'imCMS' not in pat_info:
                    self.assertEqual(pat_new[pat_info], pat_old[pat_info])
                else:
                    self.assertAlmostEqual(sum(pat_new[pat_info]),
                                           pat_old[pat_info],
                                           places=10,
                                           msg='{} vs {}'.format(
                                               sum(pat_new[pat_info]), pat_old[pat_info]))

        with open(str(new_dir / 'halo_data_wilds.json'), 'r') as jfile:
            halo_new = json.load(jfile)
        with open(str(old_dir / 'halo_data_cms.json'), 'r') as jfile:
            halo_old = json.load(jfile)

        pat_dict = dict()
        cell_type = ['Inflammation', 'Epithelium', 'Stroma']
        for pat, tmas in halo_new.items():
            if pat not in pat_dict:
                pat_dict[pat] = [[0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]]
            for tma_id in tmas:
                tma_new = tmas[tma_id]
                tma_old = halo_old[pat][tma_id]
                # print(tma_new, tma_old)
                self.assertEqual(tma_new, tma_old)
                # assert tma_new == tma_old

                if 'imCMS1' not in tma_new or tma_new['Tumor Region'] in (0, 1):
                    continue

                if float(tma_new['Tissue Area']) < 3200:
                    continue

                cell_num = list()
                for c_type in cell_type:
                    cell_nm = c_type + ' Cells'
                    cell_num.append(int(tma_new[cell_nm]))
                if sum(cell_num) < 400:
                    continue

                for i in range(4):
                    vote = tma_new['imCMS{}'.format(i + 1)]
                    pat_dict[pat][int(tma_new['Tumor Region']) -
                                  2][i] += sum(vote) / len(vote)

        labs = torch.load(str(new_dir / 'scrc_labs_wilds.pt')).numpy()
        for i in range(labs.shape[0]):
            pat = str(int(labs[i, 0]))
            t_reg = int(labs[i, 1])
            self.assertEqual(pat_dict[pat][t_reg], labs[i, -4:].tolist())

        cms = np.argmax(labs[:, -4:], axis=1)
        for i in range(3):
            lab_idx = labs[:, 1] == i
            cms_sub = cms[lab_idx]
            lab_sub = labs[lab_idx]
            pat_id = set()
            cms_tot = list()
            for j in range(lab_sub.shape[0]):
                if lab_sub[j, 0] not in pat_id:
                    pat_id.add(lab_sub[j, 0])
                    cms_tot.append(cms_sub[j])
            print(np.unique(np.asarray(cms_tot), return_counts=True))

        # visualize some imgs from pt file
        print('loading image.')
        imgs_path = new_dir / 'scrc_imgs_wilds.pt'
        imgs = torch.load(str(imgs_path)).numpy()
        print('loading image done.', imgs.shape)
        with open(str(new_dir / 'tma_pat_wilds.json'), 'r') as jfile:
            tma_pat = json.load(jfile)
            tma_pat_keys = list(tma_pat.keys())
        with open(str(old_dir / 'TMA_spots' / 'halo_data_cms.json'), 'r') as jfile:
            halo_local = json.load(jfile)

        # cnt = 0
        # for i in range(imgs.shape[0]):
        #     tma_id = tma_pat_keys[i]
        #     pat_id = tma_pat[tma_id]['patient']
        #     if pat_id in halo_local and tma_id in halo_local[pat_id]:
        #         img = Image.fromarray(np.transpose(imgs[i, 1:], axes=(1, 2, 0)), mode='RGB')
        #         img.save(old_dir/'TMA_visual'/'{}.jpg'.format(tma_id))
        #         cnt += 1
        #     if cnt >= 50:
        #         break

        pat_id = labs[:, 0]
        cms_vote = labs[:, -4:]
        df_dict = dict()
        for i in ('012', '120', '201'):
            df_path = new_dir / 'scrc_{}_wilds.csv'.format(i)
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


class test_scrc_old(unittest.TestCase):
    def __init__(self, scrc_dir, val_pat=2):
        super().__init__()

        scrc_dir = pathlib.Path(scrc_dir)
        imgs_path = scrc_dir / 'scrc_wilds_img.pt'
        imgs = torch.load(str(imgs_path))
        imgs = imgs[:, [1, 2, 3, 0]]
        ncls = imgs[:, -1]
        ncls[ncls == 3] = 4
        imgs[:, -1] = ncls
        labs_path = scrc_dir / 'scrc_wilds_lab.pt'
        labs = torch.load(str(labs_path))
        lens = [0]
        for i in range(3):
            pt_path = scrc_dir / 'scrc_symm_circle_{}.pt'.format(i)
            img, lab = torch.load(str(pt_path))
            # because of downscale the rgb value is rather a float 254.242 than 254.0
            # lead to minor diff when call byte()
            img[:, :3] = torch.clamp(img[:, :3], 0, 255)
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
    scrc = test_scrc(args.old_dir, args.new_dir)
    print('unittest scrc done!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Test the data conversion for wilds repo')
    parser.add_argument('--scrc-dir',
                        type=pathlib.Path,
                        default='/home/histopath/Data/SCRC_nuclei/TMA_spots',
                        metavar='DIR',
                        help='path the Swiss CRC dataset')
    parser.add_argument('--old-dir',
                        type=pathlib.Path,
                        default='/home/histopath/Data/SCRC_nuclei',
                        metavar='DIR',
                        help='path the Swiss CRC dataset')
    parser.add_argument('--new-dir',
                        type=pathlib.Path,
                        default='/media/histopath/Elements/jiqing/scrc_data',
                        metavar='DIR',
                        help='path the Swiss CRC dataset')
    args = parser.parse_args()

    main(args)

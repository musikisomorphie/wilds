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
from pathlib import Path
sys.path.append('.')


class test_visual_scrc:
    def __init__(self, old_dir, new_dir, img_num=200):
        old_dir = Path(old_dir)
        new_dir = Path(new_dir)
        (new_dir / 'test_tmp').mkdir(parents=True, exist_ok=True)

        with open(str(old_dir / 'halo_data_wilds.json'), 'r') as jfile:
            halo_dict = json.load(jfile)
        tma_dict = dict()
        for pat, tma_info in halo_dict.items():
            for tma in tma_info:
                if "ImageActualTif" in tma_info[tma]:
                    tma_dict[tma] = (int(pat), tma_info[tma]["ImageActualTif"])

        cnt = 0
        meta_path = new_dir / 'metadata012.csv'
        meta = pd.read_csv(str(meta_path))
        for _, row in meta.iterrows():
            print(cnt)
            img_path = new_dir / 'images' / \
                str(row['tma_reg']) / '{}_1.png'.format(row['tma_id'])
            img = Image.open(str(img_path))
            img = np.asarray(img.convert('RGB'))

            tma_id = row['tma_id'].split('_')
            tma_id = '_'.join(tma_id[:5]) + '_' + tma_id[-1]
            img_raw_path = tma_dict[tma_id][1]
            img_raw = Image.open(str(img_raw_path)).resize((256, 256))
            img_raw = np.asarray(img_raw.convert('RGB'))

            img_cat = np.concatenate([img, img_raw], axis=1)
            img = Image.fromarray(img_cat, mode='RGB')
            img.save(str(new_dir / 'test_tmp' / '{}.jpg'.format(tma_id)))
            cnt += 1
            if cnt >= img_num:
                break


class test_scrc(unittest.TestCase):
    def __init__(self, old_dir, new_dir, val_pat=2):
        super().__init__()
        new_dir = pathlib.Path(new_dir)
        _prob_id = new_dir / 'scrc_raw' / 'id.csv'
        _layout_prob = 'layout.csv'
        self.prob_id = self._proc_prob_id(_prob_id)
        self.layout_prob = self._proc_layout_prob(
            new_dir / 'scrc_raw', _layout_prob)

        self.test_cms(new_dir)
        self.test_wilds(new_dir)

    def _proc_prob_id(self, id_file):
        """Create the dictionary of prob vs id look-up table,
        this is for mapping the tma layout raw number to patient id 
        prob_id['Probenummer_Resection'] = 'ID'

        Args:
            id_file: the path to the prob_id raw file 
        """

        prob_id = dict()
        with open(str(id_file), 'r') as ifile:
            id_reader = csv.reader(ifile, delimiter=';')
            header = next(id_reader)

            for _, pat in enumerate(id_reader):
                # 'Probenummer_Resection', 'ID'
                prob, pid = pat
                dig_msg = 'the type of patient id must be integer, but got {}'. \
                    format(pid)
                self.assertTrue(pid.isdigit(), dig_msg)
                if prob != 'NA' and prob:
                    prob_id[prob] = pid

        return prob_id

    def _proc_layout_prob(self, new_dir, layout_kwd):
        """Create the dictionary of layout vs prob look-up table,
        this is for mapping the tma layout raw number to patient probe string 
        layout_prob['TMA_15_10_HE_CXXRXX'] = 'Probenummer_Resection'

        Args:
            layout_kwd: the layout keyword contained in the layout filename 
        """

        # patient id dictionary used for comparison with
        # each TMA related patient id

        layout_prob = dict()
        for layout in new_dir.glob('**/*{}'.format(layout_kwd)):
            with open(str(layout), 'r') as lfile:
                layout_reader = csv.reader(lfile, delimiter=';')
                layout_header = next(layout_reader)
                # TMA_15_10_HE
                layout_nm = (layout.name).split('.')[0]
                # This does not work for HE3 and HE2
                # Thus, comment this check
                # assert layout_header[0] in layout_nm, \
                #     'the layout filename {} should be the same to the header name {} up to H&E difference'. \
                #     format(layout_nm, layout_header[0])
                for layout_row, layout_ids in enumerate(layout_reader):
                    # ignore first two rows with controlled tma
                    if layout_row <= 1:
                        continue
                    for layout_col, layout_id in enumerate(layout_ids):
                        # ignore first two cols with controlled tma
                        if layout_col > 0:
                            col = 'C' + str(layout_col).zfill(2)
                            row = 'R' + str(layout_row + 1).zfill(2)
                            layout_prob[layout_nm + '_' +
                                        col + row] = layout_id
        return layout_prob

    def test_cms(self, new_dir):
        meta_path = new_dir / 'scrc_v1.0' / 'metadata012.csv'
        meta = pd.read_csv(str(meta_path))
        meta_dict = dict()
        pat_dict = {0: dict(), 1: dict(), 2: dict()}
        raw_pat_dict = {0: dict(), 1: dict(), 2: dict()}
        for index, row in meta.iterrows():
            tma_id = row['tma_id'].split('_')
            tma_id = '_'.join(tma_id[:5]) + '_' + tma_id[-1]
            assert tma_id not in meta_dict
            meta_dict[tma_id] = (row['cms'], row['tma_reg'], row['pat_id'])
            if row['pat_id'] not in pat_dict[row['tma_reg']]:
                pat_dict[row['tma_reg']][row['pat_id']] = row['cms']
                if row['pat_id'] not in raw_pat_dict[row['tma_reg']]:
                    raw_pat_dict[row['tma_reg']][row['pat_id']] = list()
            else:
                self.assertTrue(pat_dict[row['tma_reg']]
                                [row['pat_id']] == row['cms'])

        raw_cms_path = new_dir / 'scrc_raw' / 'SCRC_imCMS.csv'
        raw_cms = pd.read_csv(str(raw_cms_path))
        raw_cms_dict = {0: dict(), 1: dict(), 2: dict()}
        for index, row in raw_cms.iterrows():
            tma_id = row['image_ID'].split('_')
            tma_id = '_'.join(tma_id[:6])
            if tma_id in meta_dict:
                prob = self.layout_prob[tma_id]
                if prob[0] != 'B':
                    pat = prob[:-1] if prob[-1].isalpha() else prob
                else:
                    pat = self.prob_id[prob]
                # check if pat_id in metadata.csv matches raw patient id
                self.assertTrue(int(pat) == meta_dict[tma_id][-1])

                t_reg = meta_dict[tma_id][1]
                if tma_id not in raw_cms_dict[t_reg]:
                    raw_cms_dict[t_reg][tma_id] = list()
                cms_single = [float(row['imCMS{}'.format(i)])
                              for i in range(1, 5)]
                raw_cms_dict[t_reg][tma_id].append(cms_single)

        # print(raw_pat_dict)
        for t_reg, tma in raw_cms_dict.items():
            for t in tma:
                prob = self.layout_prob[t]
                if prob[0] != 'B':
                    pat = prob[:-1] if prob[-1].isalpha() else prob
                else:
                    pat = self.prob_id[prob]

                cms_mean = np.mean(np.asarray(tma[t]), axis=0).tolist()
                raw_pat_dict[t_reg][int(pat)].append(cms_mean)

        with open(str(new_dir / 'clinical_data_wilds.json'), 'r') as jfile:
            meta_new = json.load(jfile)

        cms_cnt = {0: [0, 0, 0, 0],
                   1: [0, 0, 0, 0],
                   2: [0, 0, 0, 0]}
        for t_reg, pat in raw_pat_dict.items():
            for p in pat:
                p_info = meta_new[str(p)]
                cms_sum = np.sum(np.asarray(pat[p]), axis=0)
                for i in range(4):
                    self.assertAlmostEqual(
                        float(p_info['imCMS{}'.format(i + 1)][t_reg]), float(cms_sum[i]), places=13)
                cms_cnt[t_reg][np.argmax(cms_sum)] += 1
        print(cms_cnt)

        for t_reg, tma in raw_cms_dict.items():
            for t in tma:
                prob = self.layout_prob[t]
                if prob[0] != 'B':
                    pat = prob[:-1] if prob[-1].isalpha() else prob
                else:
                    pat = self.prob_id[prob]

                cms_sum = np.sum(np.asarray(
                    raw_pat_dict[t_reg][int(pat)]), axis=0)
                if np.argmax(cms_sum) != meta_dict[t][0]:
                    print(pat, cms_sum, np.argmax(cms_sum), meta_dict[t][0])
                # self.assertTrue(np.argmax(cms_sum) == meta_dict[t][0])

    def test_wilds(self, new_dir):
        for i in (new_dir / 'scrc_v1.0' / 'images').glob('*'):
            if i.is_dir():
                for img in i.glob('**/*.png'):
                    img_id = img.stem
                    t_reg = img_id.split('_')[3]
                    if i.name == '0':
                        assert t_reg[:3] == 'III'
                    elif i.name == '1':
                        assert t_reg[:2] == 'IV'
                    else:
                        assert t_reg[:1] == 'V'

        # with open(str(new_dir / 'clinical_data_wilds.json'), 'r') as jfile:
        #     meta_new = json.load(jfile)
        # with open(str(new_dir / 'halo_data_wilds.json'), 'r') as jfile:
        #     halo_new = json.load(jfile)
        # tma_pat = dict()
        # for pat, tma_info in halo_new.items():
        #     for tma in tma_info:
        #         tma_pat[tma] = int(pat)

        for i in ('012', '120', '201'):
            df_path = new_dir / 'scrc_v1.0' / 'metadata{}.csv'.format(i)
            df = pd.read_csv(str(df_path))

            for index, row in df.iterrows():
                tma_id = row['tma_id'].split('_')
                tma_id = '_'.join(tma_id[:5]) + '_' + tma_id[-1]
                if tma_id not in self.layout_prob:
                    print(tma_id)
                    continue
                prob = self.layout_prob[tma_id]
                if prob[0] != 'B':
                    pat = prob[:-1] if prob[-1].isalpha() else prob
                else:
                    pat = self.prob_id[prob]

                # check if pat_id in metadata.csv matches raw patient id
                self.assertTrue(int(pat) == row['pat_id'])

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


class test_scrc1(unittest.TestCase):
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


class test_scrc_old1(unittest.TestCase):
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
    # scrc = test_scrc(args.old_dir, args.new_dir)
    test_visual_scrc(args.old_dir, args.new_dir)
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

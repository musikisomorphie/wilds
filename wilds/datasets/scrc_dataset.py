import os
from pathlib import Path
from collections import defaultdict

from PIL import Image
import pandas as pd
import numpy as np
import torch

from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy


class SCRCDataset(WILDSDataset):
    """
    The SCRC-WILDS dataset.
    This is a modified version of the original scrc dataset.

    Supported `split_scheme`:
        - 'official'

    Input (x):
        3-channel tma spots
        1-channel cellular annotation (optional)

    Label (y):
        y is cms class label (0-3):

    Metadata:
        Each image is annotated with its tma_id (the index of the tma spot in the whole *.pt file), 
        tma_reg (tumor front 0, micro 1, center 2),
        patient id.

    Website:
        https://www.rxrx.ai/rxrx1
        https://www.kaggle.com/c/recursion-cellular-image-classification

    Original publication:
        @inproceedings{taylor2019rxrx1,
            author = {Taylor, J. and Earnshaw, B. and Mabey, B. and Victors, M. and  Yosinski, J.},
            title = {RxRx1: An Image Set for Cellular Morphological Variation Across Many Experimental Batches.},
            year = {2019},
            booktitle = {International Conference on Learning Representations (ICLR)},
            booksubtitle = {AI for Social Good Workshop},
            url = {https://aiforsocialgood.github.io/iclr2019/accepted/track1/pdfs/30_aisg_iclr2019.pdf},
        }

    License:
        This work is licensed under a Creative Commons
        Attribution-NonCommercial-ShareAlike 4.0 International License. To view
        a copy of this license, visit
        http://creativecommons.org/licenses/by-nc-sa/4.0/.
    """
    _dataset_name = 'scrc'
    _versions_dict = {
        '1.0': {
            'download_url': None,
            'compressed_size': None}
    }

    def __init__(self,
                 version=None,
                 root_dir='data',
                 download=False,
                 split_scheme='201',
                 img_chn=[0, 1, 2, 3]):

        self._version = version
        self._split_scheme = split_scheme
        if self._split_scheme not in ['012', '120', '201']:
            raise ValueError(
                f'Split scheme {self._split_scheme} not recognized')

        # path
        self._data_dir = Path(root_dir)

        # Load splits
        df = pd.read_csv(self._data_dir /
                         'scrc_wilds_{}.csv'.format(self._split_scheme))

        self.imgs = torch.load(str(self._data_dir / 'scrc_wilds_img.pt'))
        self.imgs = self.imgs.float()
        # 3 informative classes stroma, epith, infl + 1 background
        self.imgs[:, 0] = self.imgs[:, 0].div(4.)
        # rgb pixel divided by 256
        self.imgs[:, 1:] = self.imgs[:, 1:].div(256.)
        self.imgs = self.imgs[:, img_chn]
        # Training:   the tma spots related to the 'xx' tumor region
        #             embedded in 'xxz' of split_scheme
        # Validation: the tma spots realted to the 'z' of 'xxz'
        #             randomly sampled 2 tma spots/patient
        # Test:       the rest of tma spots related to the 'z' of 'xxz'
        self._split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        self._split_names = {
            'train': 'Train',
            'val': 'Validation',
            'test': 'Test'
        }

        self._split_array = df.dataset.apply(self._split_dict.get).values
        self._input_array = df['tma_id'].values
        # Labels
        self._y_array = torch.tensor(df['cms'].values)
        self._n_classes = max(df['cms']) + 1
        self._y_size = 1
        assert len(np.unique(df['cms'])) == self._n_classes

        self._metadata_array = torch.tensor(
            np.stack([df['tma_reg'].values,
                      df['pat_id'].values,
                      self.y_array], axis=1)
        )
        self._metadata_fields = ['tma_reg', 'pat_id', 'y']

        # eval grouper
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['y'])
        )

        super().__init__(root_dir, download, split_scheme)

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are
                predicted labels (LongTensor). But they can also be other model
                outputs such that prediction_fn(y_pred) are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metric = Accuracy(prediction_fn=prediction_fn)
        return self.standard_group_eval(
            metric,
            self._eval_grouper,
            y_pred, y_true, metadata)

    def get_input(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """
        # All images are in the train folder
        img = self.imgs[self._input_array[idx]]
        return img

import os
import pickle

import cv2
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from .paths_definitions import (
    DIR_GEOMETRIC_TRAIN_LMDB,
    DIR_GEOMETRIC_VAL_TEST_LMDB,
    DIR_TEMPORAL_TRAIN_LMDB,
    DIR_TEMPORAL_VAL_TEST_LMDB,
)

from .utils import utils_dataset as utils_dataset


class LMDBDataset(Dataset):
    def __init__(self,
                 fold_number,
                 type_dataset,
                 load_mode,
                 transformations=None,
                 balance=False,
                 max_samples=None,
                 create_empty=False,
                 alignment='geometric'):

        if create_empty:
            self.lmdb_path = ''
            self.type_dataset = None
            self.fold_number = None
            self.transformations = None
            self.load_mode = None
            self.keys_ds = []
            self.db = None
            self.half_window_size = None
            return

        self.db = None  # lazy init db for pickle

        # LMDB é um diretório
        assert type_dataset in ['train', 'validation', 'test']
        assert alignment in ['geometric', 'temporal']
        if alignment == 'geometric':
            self.lmdb_path = DIR_GEOMETRIC_TRAIN_LMDB if type_dataset == 'train' else DIR_GEOMETRIC_VAL_TEST_LMDB
        elif alignment == 'temporal':
            self.lmdb_path = DIR_TEMPORAL_TRAIN_LMDB if type_dataset == 'train' else DIR_TEMPORAL_VAL_TEST_LMDB
        print(self.lmdb_path)
        assert os.path.isdir(self.lmdb_path)
        assert load_mode in ['block', 'keyframe']

        # If train and load_mode=>'keyframe', central window (half_window_size) is the central frame of the batch (7)
        # If train and load_mode=>'block', it won't use the central frame window (half_window_size)
        # If val or test, load_mode=>'keyframe' each batch has a single frame only :. central frame (half_window_size) é o is the frame itself
        self.half_window_size = 7 if type_dataset == 'train' else 0

        self.type_dataset = type_dataset
        self.fold_number = fold_number
        self.transformations = transformations
        self.load_mode = load_mode

        # Load auxiliary data only
        self.keys_ds = utils_dataset.get_keys_lmdb(fold_number,
                                                   lmdb_path=self.lmdb_path,
                                                   type_dataset=type_dataset)
        if balance:
            self.keys_ds = utils_dataset.balance_reduce_shuffle_keys_lmdb(self.keys_ds, max_samples)

    def decode(self, data):
        data = pickle.loads(data)
        refs = np.array([cv2.imdecode(ref, cv2.IMREAD_UNCHANGED) for ref in data["refs"]])
        tars = np.array([cv2.imdecode(tar, cv2.IMREAD_UNCHANGED) for tar in data["tars"]])
        bboxes = data["bboxes"]
        classes = data["classes"]
        return refs, tars, bboxes, classes

    def __len__(self):
        return len(self.keys_ds)  # batches

    def __getitem__(self, idx):
        if self.db is None:
            self.db = lmdb.open(self.lmdb_path, readonly=True, lock=False)

        key = self.keys_ds[idx]['lmdb_key']

        with self.db.begin(write=False) as tx:
            buff = tx.get(key)

        ref, tar, bboxes, classes = self.decode(buff)
        # Apply transformations
        ref = torch.stack([self.transformations(r) for r in ref])
        tar = torch.stack([self.transformations(t) for t in tar])
        # Keep only frames that are according to the load_mode
        if self.load_mode == 'keyframe':
            # Keep only the central frame (half_window_size)
            ref = ref[self.half_window_size]
            tar = tar[self.half_window_size]
            bboxes = bboxes[self.half_window_size]
            classes = classes[self.half_window_size]
            assert classes == self.keys_ds[idx]['class_keyframe']
            return ref, tar, classes, bboxes
        # If blocking mode, get all frames within the block
        elif self.load_mode == 'block':
            return ref, tar, classes, bboxes

    def clone(self):
        ret = LMDBDataset.create_empty()
        ret.lmdb_path = self.lmdb_path
        ret.type_dataset = self.type_dataset
        ret.type_dataset = self.type_dataset
        ret.fold_number = self.fold_number
        ret.load_mode = self.load_mode
        ret.transformations = self.transformations
        ret.keys_ds = self.keys_ds
        ret.db = self.db
        ret.half_window_size = self.half_window_size
        return ret

    def get_objects(self):
        return list(set([el['object'] for el in self.keys_ds]))

    @staticmethod
    def create_empty():
        return LMDBDataset('', '', '', create_empty=True)

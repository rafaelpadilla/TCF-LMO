# Dataset to load frames aligned geometricaly

import os
import pickle
import random

import lmdb

from .utils_functions import get_files_recursively
from .VDAO_folds.hdf5dataset_v2 import HDF5Dataset

objects_folds_prop_5_3 = {
    1: {
        'train': ['brown box', 'camera box', 'dark-blue box', 'shoe', 'towel'],
        'validation': ['black backpack', 'pink bottle', 'white jar'],
        'test': ['black coat']
    },
    2: {
        'train': ['black coat', 'camera box', 'dark-blue box', 'pink bottle', 'towel'],
        'validation': ['brown box', 'shoe', 'white jar'],
        'test': ['black backpack']
    },
    3: {
        'train': ['black coat', 'black backpack', 'pink bottle', 'shoe', 'white jar'],
        'validation': ['camera box', 'dark-blue box', 'towel'],
        'test': ['brown box']
    },
    4: {
        'train': ['black coat', 'black backpack', 'pink bottle', 'towel', 'white jar'],
        'validation': ['brown box', 'dark-blue box', 'shoe'],
        'test': ['camera box']
    },
    5: {
        'train': ['brown box', 'pink bottle', 'shoe', 'towel', 'white jar'],
        'validation': ['black coat', 'black backpack', 'camera box'],
        'test': ['dark-blue box']
    },
    6: {
        'train': ['black coat', 'black backpack', 'brown box', 'camera box', 'towel'],
        'validation': ['dark-blue box', 'shoe', 'white jar'],
        'test': ['pink bottle']
    },
    7: {
        'train': ['black backpack', 'dark-blue box', 'pink bottle', 'towel', 'white jar'],
        'validation': ['black coat', 'brown box', 'camera box'],
        'test': ['shoe']
    },
    8: {
        'train': ['black coat', 'brown box', 'dark-blue box', 'shoe', 'white jar'],
        'validation': ['black backpack', 'camera box', 'pink bottle'],
        'test': ['towel']
    },
    9: {
        'train': ['black backpack', 'brown box', 'camera box', 'dark-blue box', 'shoe'],
        'validation': ['black coat', 'pink bottle', 'towel'],
        'test': ['white jar']
    },
}


def get_file_paths_geo_aligned_hdf5_files(type_dataset, type_alignment, objects=None):
    if type_dataset in ['validation', 'test']:
        dir_to_read = lmdb.DIR_FRAMES_VAL_TEST
    elif type_dataset == 'train':
        dir_to_read = lmdb.DIR_FRAMES_TRAIN
    ret = []
    # If specific objects are defined
    if objects is not None and objects is not []:
        objs_names = [o.replace(' ', '_').replace('-', '_') for o in objects]
        for o in objs_names:
            ret += [get_files_recursively(os.path.join(dir_to_read, o), '*.hd5')]
        ret = sum(ret, [])  # Flatten list
    else:
        # Get all hdf5 files from the dataset
        ret = get_files_recursively(dir_to_read, '*.hd5')
    return ret


def get_keys_lmdb(fold_number, lmdb_path, type_dataset='train'):
    assert type_dataset in [
        'train', 'validation', 'test'
    ], 'Argument type_dataset must be either \'train\', \'validation\' or \'test\''

    # proportion training(5) : validation(3) : testing(1)
    # define here the validation, training and testing objects of each fold
    objects_fold = objects_folds_prop_5_3[fold_number][type_dataset]
    objects_fold = [obj.replace('-', '_').replace(' ', '_') for obj in objects_fold]
    # Open lmdb and get list that contains the keys of the objects, batches, classes, etc
    with lmdb.open(lmdb_path, readonly=True, lock=False) as env:
        with env.begin() as tx:
            type_ds_str = 'validation' if type_dataset in ['validation', 'test'] else type_dataset
            key_buff = tx.get(f"{type_ds_str}_keys".encode("ascii"))
            list_data = pickle.loads(key_buff)

    return [d for d in list_data if d['object'] in objects_fold]


def balance_reduce_shuffle_keys_lmdb(hdf5_dataset, max_samples=None, quiet=True):
    # Balance and reduce
    pos_frames = [b for b in hdf5_dataset if b['class_keyframe'] is True]
    neg_frames = [b for b in hdf5_dataset if b['class_keyframe'] is False]
    assert len(pos_frames) + len(neg_frames) == len(hdf5_dataset)
    if max_samples is None:
        max_samples = len(hdf5_dataset)
    qty_to_balance = min(len(pos_frames), len(neg_frames), max_samples // 2)
    pos_frames = pos_frames[0:qty_to_balance]
    neg_frames = neg_frames[0:qty_to_balance]
    if not quiet:
        print(f'{len(pos_frames)} positive samples')
        print(f'{len(neg_frames)} negative samples')
    res = pos_frames + neg_frames
    # Shuffle
    random.seed(123)
    random.shuffle(res)
    return res


def get_VDAO_dataset(fold_number, type_dataset='train'):
    assert type_dataset in [
        'train', 'validation', 'test'
    ], 'Argument type_dataset must be either \'train\', \'validation\' or \'test\''

    base_dir = DIR_FRAMES_TRAIN if type_dataset == 'train' else DIR_FRAMES_VAL_TEST
    # proportion training(5) : validation(3) : testing(1)
    fold_info = {'objects': objects_folds_prop_5_3[fold_number]}
    hdf5_file_dirs = []
    for obj in fold_info['objects'][type_dataset]:
        obj = obj.replace('-', '_').replace(' ', '_')
        dir_obj = os.path.join(base_dir, obj)
        assert os.path.isdir(dir_obj)
        hdf5_file_dirs.append(dir_obj)

    hdf5_datasets = [HDF5Dataset(hdf5_dir, type_ds=type_dataset) for hdf5_dir in hdf5_file_dirs]

    # after loading all datasets, gather in a unique dataset all the other datasets
    _id = None
    _id = [idx for idx, ds in enumerate(hdf5_datasets) if len(ds.blocks)]
    assert len(_id) != 0
    _id = _id[0]
    for idx, ds in enumerate(hdf5_datasets):
        if idx == _id or len(ds.blocks) == 0:
            continue
        assert hdf5_datasets[_id].half_window_size == ds.half_window_size
        assert hdf5_datasets[_id].load_mode == ds.load_mode
        assert hdf5_datasets[_id].type_ds == ds.type_ds
        hdf5_datasets[_id].blocks += ds.blocks
        hdf5_datasets[_id].files_loaded = {**hdf5_datasets[_id].files_loaded, **ds.files_loaded}
        hdf5_datasets[_id].hdf5_file = {**hdf5_datasets[_id].hdf5_file, **ds.hdf5_file}
    return hdf5_datasets[_id]


def balance_reduce_shuffle_samples(hdf5_dataset, max_samples=None):
    # Balance and reduce
    pos_frames = [b for b in hdf5_dataset.blocks if b[2] is True]
    neg_frames = [b for b in hdf5_dataset.blocks if b[2] is False]
    assert len(pos_frames) + len(neg_frames) == len(hdf5_dataset.blocks)
    if max_samples is None:
        max_samples = len(hdf5_dataset.blocks)
    qty_to_balance = min(len(pos_frames), len(neg_frames), max_samples // 2)
    pos_frames = pos_frames[0:qty_to_balance]
    neg_frames = neg_frames[0:qty_to_balance]
    print(f'{len(pos_frames)} amostras positivas')
    print(f'{len(neg_frames)} amostras negativas')
    hdf5_dataset.blocks = pos_frames + neg_frames
    # Shuffle
    random.seed(123)
    random.shuffle(hdf5_dataset.blocks)
    return hdf5_dataset


def split_data_set_into_videos(unique_ds):
    ret = []
    for vid in unique_ds.hdf5_file.keys():
        hdf5_ds = HDF5Dataset.create_empty()
        hdf5_ds.blocks = [block for block in unique_ds.blocks if block[0] == vid]
        hdf5_ds.files_loaded = {vid: unique_ds.files_loaded[vid]}
        hdf5_ds.half_window_size = unique_ds.half_window_size
        hdf5_ds.hdf5_file = {vid: unique_ds.hdf5_file[vid]}
        hdf5_ds.load_mode = unique_ds.load_mode
        hdf5_ds.type_ds = unique_ds.type_ds
        ret.append(hdf5_ds)
    return ret


def split_data_set_into_videos_lmdb(unique_ds):
    ret = []
    videos = set([k['video_name'] for k in unique_ds.keys_ds])
    for vid in videos:
        keys = [k for k in unique_ds.keys_ds if k['video_name'] == vid]
        keys = sorted(keys, key=lambda x: x['batch'])
        # Makes sure it is ordered by batch
        for i, k in enumerate(keys):
            if i != 0:
                assert k['batch'] == keys[i - 1]['batch'] + 1
        # Makes a copy of the original dataset, changing its keys_ds
        new_ds = unique_ds.clone()
        new_ds.keys_ds = keys
        ret.append(new_ds)
    return ret

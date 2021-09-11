# Esta versão foi criada em 4 de março de 2021 que é adapdata para carregar os
# arquivos hdf5 com o alinhamento geométrico feito com blocos, onde é preciso
# saber quem é keyframe e quem não é.
# Nesta adaptação, ao invés de guardar todos os objetos (ref, tar, label, bb, etc)
# em uma lista comum. Vou guardar em dicionários cuja chave é o id

import ast
import fnmatch
import os
import pickle
import time
from pathlib import Path

import h5py
import more_itertools as mit
import numpy as np
import torch
from PIL import Image
from torch.utils import data

from .utils import get_files


class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.

    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
    """
    def __init__(self,
                 dir_hdf5_files,
                 type_ds,
                 load_mode='keyframe',
                 raise_errors=False,
                 create_empty=False):
        super().__init__()
        self.blocks = []
        self.files_loaded = {}
        self.hdf5_file = {}

        if create_empty:
            self.load_mode = ''
            self.type_ds = ''
            return

        self.type_ds = 'validation' if type_ds.upper() == 'VAL' else type_ds.lower()
        assert load_mode in ['keyframe', 'block']
        self.load_mode = load_mode

        assert os.path.isdir(dir_hdf5_files)
        basename = os.path.basename(dir_hdf5_files)
        fp_to_save = os.path.join(dir_hdf5_files, f'{basename}.pickle')
        # Verifica a existência de arquivo pickle já carregado
        if os.path.isfile(fp_to_save):
            print(f'Reading pre-loaded file {fp_to_save}')
            dict_read = pickle.load(open(fp_to_save, 'rb'))
            self.load_mode = dict_read['load_mode']
            self.half_window_size = dict_read['half_window_size']
            self.blocks = dict_read['blocks']
            self.type_ds = dict_read['type_ds']
            self.files_loaded = dict_read['files_loaded']
            self.hdf5_file = {bn: h5py.File(fp, 'r') for bn, fp in self.files_loaded.items()}
            # Neste pickle não tem a classe do keyframe :. Será incluida.
            if len(self.blocks[0]) == 2:
                new_blocks = []
                for vid, data in self.hdf5_file.items():
                    kf_labels = {
                        int(type_data.replace('label_sample_', '')):
                        torch.tensor(self.hdf5_file[vid][type_data]).item()
                        for type_data, dd in data.items() if 'label' in type_data
                    }
                    if self.half_window_size == 0:
                        new_blocks += [(block[0], block[1], kf_labels[block[1][0]])
                                       for block in self.blocks if block[0] == vid]
                    else:
                        new_blocks += [(block[0], block[1],
                                        kf_labels[block[1][self.half_window_size]])
                                       for block in self.blocks if block[0] == vid]
                self.blocks = new_blocks
                # Substitui o arquivo pickle com um novo, onde os blocks terão a classe
                dict_to_save = {
                    # 'path': dir_hdf5_files,
                    'load_mode': self.load_mode,
                    'half_window_size': self.half_window_size,
                    'blocks': self.blocks,
                    'type_ds': self.type_ds,
                    'files_loaded': self.files_loaded
                }
                fp_to_save = os.path.join(dir_hdf5_files, f'{basename}.pickle')
                pickle.dump(dict_to_save, open(fp_to_save, 'wb'))
            return

        print(f'Pre-loaded file not found: {basename}.pickle')
        print(f'Loading dataset from {dir_hdf5_files}')

        # Search for all hd5 files
        files = get_files(dir_hdf5_files, 'hd5', recursive=True)
        if len(files) == 0:
            if raise_errors:
                raise RuntimeError('No hdf5 datasets found')
            else:
                print('No hdf5 datasets found')
            return

        # Processa cada arquivo hd5
        for h5dataset_fp in files:
            self._add_data_infos(h5dataset_fp)

        # Salva em um dicionário os arquivos que acabaram de ser carregados
        dict_to_save = {
            'path': dir_hdf5_files,
            'load_mode': load_mode,
            'half_window_size': self.half_window_size,
            'blocks': self.blocks,
            'type_ds': self.type_ds,
            'files_loaded': self.files_loaded
        }
        pickle.dump(dict_to_save, open(fp_to_save, 'wb'))
        print(f'Saving pre-loaded file {fp_to_save} with {len(self.blocks)} samples.')

    def get_biggest_bb(self, bbs):
        if bbs.nelement() == 0:
            return torch.tensor([0, 0, 0, 0])
            # O data loader nao pode receber objeto None
            # batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'NoneType'>
            return None

        largest_area = -1
        selected_bb = None
        for bb in bbs.squeeze(0):
            bb = bb.squeeze()
            area = (bb[2] - bb[0]) * (bb[3] - bb[1])
            if area > largest_area:
                largest_area = area
                selected_bb = bb
        return selected_bb

    def __getitem__(self, index):
        # start = time.time()
        # # get reference frames
        # ref = self.get_data("img_ref_sample_", index)
        # tar = self.get_data("img_tar_sample_", index)
        # # load as torch arrays
        # ref = torch.tensor(ref)
        # tar = torch.tensor(tar)
        # # get label
        # label = self.get_data("label_sample_", index)
        # label = torch.tensor(label).type(torch.float)
        # # get bb
        # bbs = self.get_data("bb_sample_", index)
        # # replace cases where bb => tensor([]), which means no bounding box (no anomaly frames) by bb [0,0,0,0]
        # bbs = [[[[0, 0, 0, 0]]] if len(el.shape) == 1 else el for i, el in enumerate(bbs)]
        # bbs = torch.tensor(bbs)
        # if self.load_mode == 'block':
        #     bb = torch.tensor([self.get_biggest_bb(bb).tolist() for bb in bbs])
        # elif self.load_mode == 'keyframe':
        #     # return self.hdf5_file[_filename][f'{type_data}{sample_id}']
        #     bb = self.get_biggest_bb(bbs).unsqueeze(0)
        # end = time.time() - start
        # return ref, tar, label, bb

        # start = time.time()
        ret = self.get_multiple_data(
            ['img_ref_sample_', 'img_tar_sample_', 'label_sample_', 'bb_sample_'], index)

        ref2 = torch.tensor(np.array(ret['img_ref_sample_']))  #0.74s
        tar2 = torch.tensor(np.array(ret['img_tar_sample_']))

        # ref2 = torch.as_tensor(np.array(ret['img_ref_sample_'])) 0.76s
        # tar2 = torch.as_tensor(np.array(ret['img_tar_sample_']))

        # ref2 = torch.tensor(ret['img_ref_sample_']) 2.24s
        # tar2 = torch.tensor(ret['img_tar_sample_'])
        label2 = torch.tensor(np.array(ret['label_sample_'])).type(torch.float)
        # replace cases where bb => tensor([]), which means no bounding box (no anomaly frames) by bb [0,0,0,0]
        bbs2 = [[[[0, 0, 0, 0]]] if len(el.shape) == 1 else el
                for i, el in enumerate(ret['bb_sample_'])]
        bbs2 = torch.tensor(np.array(bbs2))
        if self.load_mode == 'block':
            bb2 = torch.tensor([self.get_biggest_bb(bb).tolist() for bb in bbs2])
        elif self.load_mode == 'keyframe':
            # return self.hdf5_file[_filename][f'{type_data}{sample_id}']
            bb2 = self.get_biggest_bb(bbs2).unsqueeze(0)
        # end2 = time.time() - start
        # print(f'{end2} s')

        # assert (ref == ref2).all()
        # assert (tar == tar2).all()
        # assert (label == label2).all()
        # assert (bb == bb2).all()
        # print(end2 - end, end2 < end)

        # return ref2, tar2, label2, bb2

        return ref2, tar2, label2, bb2, self.blocks[index]

    def __len__(self):
        return len(self.blocks)

    def _add_data_infos(self, file_path):
        fp_basename = os.path.splitext(os.path.basename(file_path))[0]

        self.hdf5_file[fp_basename] = h5py.File(file_path, 'r')
        self.files_loaded[fp_basename] = file_path

        dict_sampleid_tarframe = {}
        key_frames_sampleid = []
        # Associa um sample_id (0,1,2...) com o id do frame target
        # Obtém os ids dos frames targets que são keyframes
        selected = self.hdf5_file[fp_basename]
        for gname, group in selected.items():
            if 'extra_info' in gname:
                sample_id = int(gname.replace('extra_info_', ''))
                # assert sample_id not in dict_sampleid_tarframe
                frame_id = ast.literal_eval(group[()])['geo_align'][1]
                dict_sampleid_tarframe[sample_id] = frame_id
            elif 'keyframe' in gname and group[()] == True:
                sample_id = int(gname.replace('keyframe', ''))
                key_frames_sampleid.append(sample_id)
        # Certifica que os frames alvos só tenham 1 amostra
        assert len(set(dict_sampleid_tarframe.values())) == len(dict_sampleid_tarframe.values())

        # Obtém uma lista com os ids dos target frames que são keyframes
        keyframes_tar = [dict_sampleid_tarframe[k] for k in key_frames_sampleid]
        keyframes_tar = sorted(keyframes_tar, key=lambda k: k)
        # Obtém uma lista com os ids dos  target frames que não são keyframes
        non_keyframes_tar = [k for k in dict_sampleid_tarframe.values() if k not in keyframes_tar]
        non_keyframes_tar = sorted(non_keyframes_tar, key=lambda k: k)
        # Se for dataset de validação ou teste, todos frames são keyframes
        if self.type_ds in ['validation', 'test']:
            assert non_keyframes_tar == []
            # qte frames à esquerda e/ou à direita do keyframe
            self.half_window_size = 0
        else:
            groups = [list(group) for group in mit.consecutive_groups(non_keyframes_tar)]
            # qte frames à esquerda e/ou à direita do keyframe
            self.half_window_size = min([len(g) for g in groups])
        # Obtém um dicionário inverso tar_frameid -> sampleid
        dict_tarframe_sampleid = {v: k for k, v in dict_sampleid_tarframe.items()}
        # Obtém os blocks = listas identificando os frames targets alinhados geometricamente
        _blocks_tar = [
            list(np.arange(k - self.half_window_size, k + self.half_window_size + 1))
            for k in keyframes_tar
        ]
        # Cria lista com os blocos associados aos samples_ids
        for block in _blocks_tar:
            self.blocks.append((fp_basename, [dict_tarframe_sampleid[b] for b in block]))
        # Certifica que em cada bloco é formado por 15 pares de frames e existe somente 1 keyframe
        if self.type_ds == 'train':
            len_blocks = 15
        elif self.type_ds in ['validation', 'test']:
            len_blocks = 1
        for b in self.blocks:
            assert len(b[1]) == len_blocks

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type_data, idx):
        _block = self.blocks[idx]
        _filename, _frames_samples = _block[0], _block[1]
        # Se for modo keyframe, pega só o keyframe (cuja posição está no self.half_window_size)
        if self.load_mode == 'keyframe':
            sample_ids = [_frames_samples[self.half_window_size]]
        # Se for modo 'block', pega todos os frames do bloco
        elif self.load_mode == 'block':
            sample_ids = _frames_samples

        return [self.hdf5_file[_filename][f'{type_data}{sample_id}'] for sample_id in sample_ids]

    def get_multiple_data(self, list_type_data, idx):
        _block = self.blocks[idx]
        _filename, _frames_samples = _block[0], _block[1]
        # Se for modo keyframe, pega só o keyframe (cuja posição está no self.half_window_size)
        if self.load_mode == 'keyframe':
            sample_ids = [_frames_samples[self.half_window_size]]
        # Se for modo 'block', pega todos os frames do bloco
        elif self.load_mode == 'block':
            sample_ids = _frames_samples

        a = self.hdf5_file[_filename]
        res = {}
        for type_data in list_type_data:
            res[type_data] = [a[f'{type_data}{sample_id}'] for sample_id in sample_ids]
        return res

    def clone(self):
        ret = HDF5Dataset.create_empty()
        ret.blocks = self.blocks.copy()
        ret.files_loaded = self.files_loaded.copy()
        ret.hdf5_file = self.hdf5_file.copy()
        ret.load_mode = self.load_mode
        ret.type_ds = self.type_ds
        ret.half_window_size = self.half_window_size
        return ret

    @staticmethod
    def create_empty():
        return HDF5Dataset('', '', create_empty=True)


class iHDF5Dataset(HDF5Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        # ref, tar, label = super().__getitem__(index)
        return super().__getitem__(index)

        return {'ref_tensor': ref, 'tar_tensor': tar}, label

import math

import torch
import torch.nn as nn


class TemporalConsistency():
    def __init__(self, max_frames=15, type_voting='unanimity'):
        assert type_voting in ['majority', 'unanimity']
        self.type_voting = type_voting
        self.voting_window = max_frames
        self.max_frames = max_frames
        self.frames_train = None
        self.count_frames_train = 0
        # Lista que guarda os frames usados
        self.frames_inference = None
        self.gt_labels_inference = []
        self.gt_bbs_inference = []
        self.count_frames_inference = 0

    ################################################################################################
    # TRAINING METHODS
    ################################################################################################
    def start_new_train_batch(self):
        self.frames_train = None
        self.count_frames_train = 0

    def gather_one_by_one_train(self, frame):
        assert self.count_frames_train < self.max_frames
        if self.frames_train is None:
            h, w = frame.shape
            self.frames_train = torch.zeros((self.max_frames, h, w),
                                            dtype=torch.float32,
                                            device=frame.device)
        if len(self.frames_train) != 0:
            assert frame.shape == self.frames_train[0, :, :].shape
        self.frames_train[self.count_frames_train] = frame
        self.count_frames_train += 1

    def gather_many_train(self, frames):
        assert self.count_frames_train < self.max_frames
        for frame in frames:
            # self.gather_one_by_one_train(frame)
            self.gather_one_by_one_train(frame.squeeze())

    def train(self):
        if self.type_voting == 'unanimity':
            return self.train_unanimity()
        else:  # majority
            return self.train_majority()

    def train_majority(self):
        res_dict = {}
        # Garante que todos frames foram adicionados
        h, w = self.frames_train[0].shape[-2:]
        # Aplica um threshold de 0.5 para deixar pixels binários
        thresholded = (self.frames_train > 0.5) * 1
        # Frame central
        mid_id = self.max_frames // 2
        for i in range(mid_id + 1):
            ini, end = mid_id - i, mid_id + i + 1
            window = end - ini
            half_window = math.ceil(window / 2)
            # calcula se maioria dos pixels estão acesos dentro da janela
            consistency = (thresholded[ini:end, :, :].sum(axis=0) >= half_window) * 1.
            res_dict[window] = consistency
        return res_dict

    def train_unanimity(self):
        res_dict = {}
        # Garante que todos frames foram adicionados
        h, w = self.frames_train[0].shape[-2:]
        # Aplica um threshold de 0.5 para deixar pixels binários
        thresholded = (self.frames_train > 0.5) * 1
        # Frame central
        mid_id = self.max_frames // 2
        for i in range(mid_id + 1):
            ini, end = mid_id - i, mid_id + i + 1
            window = end - ini
            # calcula se todos pixels acesos estão dentro da janela
            consistency = (thresholded[ini:end, :, :].sum(axis=0) == window) * 1.
            res_dict[window] = consistency
        return res_dict

    ####################################################################################################################
    # INFERENCE
    ####################################################################################################################
    def start_new_inference_batch(self):
        self.frames_inference = None
        self.count_frames_inference = 0

    def set_batch_info(self, total_samples, samples_to_accumulate):
        self.batch_sizes = []
        for i in range(total_samples // samples_to_accumulate):
            self.batch_sizes.append(samples_to_accumulate)
        rest = total_samples % samples_to_accumulate
        if rest != 0:
            self.batch_sizes.append(rest)
        assert sum(self.batch_sizes) == total_samples

    def gather_one_by_one_inference(self, frame):
        if self.frames_inference is None:
            h, w = frame.shape
            self.frames_inference = torch.zeros((self.batch_sizes[0], h, w))
        self.frames_inference[self.count_frames_inference] = frame
        self.count_frames_inference += 1

    def gather_gt_label_inference(self, label):
        self.gt_labels_inference.append(label)

    def gather_gt_bb_inference(self, bb):
        self.gt_bbs_inference.append(bb)

    def inference_validation_test_unanimity(self, x):
        # Apply threshold of 0.5 to make binary pixels
        thresholded = (x > 0.5) * 1
        # Compute if all pixels are within the temporal window
        # Obs: For testing and validation, the batch size is the window size
        window = len(x)
        consistency = (thresholded[:, :, :].sum(axis=0) == window) * 1.
        return consistency

    def inference_validation_test_majority(self, x):
        # Apply threshold of 0.5 to make binary pixels
        thresholded = (x > 0.5) * 1
        # Compute if all pixels are within the temporal window
        # Obs: For testing and validation, the batch size is the window size
        window = len(x)
        half_window = math.ceil(window / 2)
        consistency = (thresholded[:, :, :].sum(axis=0) >= half_window) * 1.
        return consistency

    def inference_validation_test(self, x):
        if self.type_voting == 'unanimity':
            return self.inference_validation_test_unanimity(x)
        else:  # majority
            return self.inference_validation_test_majority(x)

    def is_buffer_complete(self):
        # Verifica se já tem um batch completado de amostras
        return self.batch_sizes[0] == self.count_frames_inference

    def clean_buffer(self):
        self.frames_inference = None
        self.gt_labels_inference = []
        self.gt_bbs_inference = []
        self.count_frames_inference = 0

    def inference_train_unanimity(self, x):
        assert len(x) == self.max_frames
        # Aplica um threshold de 0.5 para deixar pixels binários
        thresholded = (x > 0.5) * 1
        # Frame central
        mid_id = self.max_frames // 2
        neighbor = (self.voting_window - 1) // 2
        ini, end = mid_id - neighbor, mid_id + neighbor + 1
        window = end - ini
        assert window == self.voting_window
        # calcula se todos pixels acesos estão em dentro da janela
        consistency = (thresholded[ini:end, :, :].sum(axis=0) == window) * 1.
        return consistency

    def inference_train_majority(self, x):
        assert len(x) == self.max_frames
        # Aplica um threshold de 0.5 para deixar pixels binários
        thresholded = (x > 0.5) * 1
        # Frame central
        mid_id = self.max_frames // 2
        neighbor = (self.voting_window - 1) // 2
        ini, end = mid_id - neighbor, mid_id + neighbor + 1
        window = end - ini
        assert window == self.voting_window
        half_window = math.ceil(window / 2)
        # calcula se maioria dos pixels estão acesos dentro da janela
        consistency = (thresholded[ini:end, :, :].sum(axis=0) >= half_window) * 1.
        return consistency

    def inference_train(self, x):
        if self.type_voting == 'unanimity':
            return self.inference_train_unanimity(x)
        else:  # majority
            return self.inference_train_majority(x)

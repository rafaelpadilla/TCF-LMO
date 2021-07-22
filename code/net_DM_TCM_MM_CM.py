import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
from modules.modules_functions import (ChangeDynamicRange, ChangeDynamicRangeEuclideanDistance,
                                       LearnableSigmoid, N_TensorsDifference, SumAllPixels)
from modules.morphology_sigmoid import Closing, Opening
from modules.platan_v6 import PLAtan
from modules.temporal_consistency import TemporalConsistency


class Platannet(nn.Module):
    def __init__(self, net_params, device=None):
        super(Platannet, self).__init__()
        self.device = device
        # DM
        distance_params = net_params['distance_module']
        self.dissimilarity_module = N_TensorsDifference(
            n_pairs=distance_params['pairs'], gamma_sigmoid=distance_params['gamma_bin_sigmoid'])
        # TCM
        self.temporal_consistency = TemporalConsistency(
            max_frames=net_params['temporal_consistency_module']['max_frames'],
            type_voting=net_params['temporal_consistency_module']['type'])
        # MM
        morph_params = net_params['morphology_module']
        self.max_radius = morph_params['max_radius']
        self.opening = Opening(
            se_gamma=morph_params['open']['se_gamma'],
            se_radius=morph_params['open']['radius'],
            kwargs_thresh_params=morph_params['open']['params_thresh'],
            epsilon=morph_params['open']['epsi'],
        )
        self.closing = Closing(
            se_gamma=morph_params['close']['se_gamma'],
            se_radius=morph_params['close']['radius'],
            kwargs_thresh_params=morph_params['close']['params_thresh'],
            epsilon=morph_params['close']['epsi'],
        )
        # Sum all pixels on
        self.sum_pixels_on = SumAllPixels()
        # Change dynamic range
        scale_params = net_params['scale_module']
        self.change_scale_for_classification = ChangeDynamicRange(0, scale_params['total_pixels'],
                                                                  scale_params['min'],
                                                                  scale_params['max'])
        # CM
        classification_params = net_params['classification_module']
        if classification_params['function'] == 'platan':
            self.classification_function = PLAtan(beta=classification_params['beta'],
                                                  min_allowed_thresh=0.,
                                                  max_allowed_thresh=1.,
                                                  threshold=classification_params['thresh'])
        elif classification_params['function'] == 'sigmoid':
            self.classification_function = LearnableSigmoid(
                threshold=classification_params['thresh'], gamma=classification_params['gamma'])

    def inference_validation_test(self, tensors):
        # batch, 256, 81, 144
        out = self.dissimilarity_module(tensors['feat_ref'], tensors['feat_tar'])
        # batch, 81, 144
        if out.dim() == 2:
            out = out.unsqueeze(0)
        # Apply temporal consistency
        out = self.temporal_consistency.inference_validation_test(out)
        # 81, 144
        # MM (opening)
        out = self.opening(out)
        # 1, 1, 81, 144
        # MM (closing)
        out = self.closing(out)
        # 1, 1, 81, 144
        # Count pixels "on" in each image
        out = self.sum_pixels_on(out)
        # 1, 1
        # Now we have the amount of pixels on, we will transform it linearly to have a range between 0 and 1
        out = self.change_scale_for_classification(out)
        # 1, 1
        # CM
        out = self.classification_function(out)
        # 1, 1
        return out

    def forward(self, tensors):
        # DM
        out = self.dissimilarity_module(tensors['feat_ref'], tensors['feat_tar'])
        # batch, 81, 144
        assert out.min().item() >= 0.
        assert out.max().item() <= 1.
        # Train DM
        if tensors['cycle_name'] == 'training DM':
            if len(out.shape) == 2:
                return out.unsqueeze(0).unsqueeze(0)
            else:
                return out.unsqueeze(1)
        # Training TCM
        if tensors['cycle_name'] == 'training TCM':
            self.temporal_consistency.gather_many_train(out)
            # return a dictionary with tensors
            return self.temporal_consistency.train()
        # During training of MM and CM, apply temporal consistency and store frames til we have a full batch
        if tensors['cycle_name'] in ['training MM', 'training CM']:
            # Apply temporal consistency
            # 15, 81, 144
            out = self.temporal_consistency.inference_train(out)
            # 81, 144
            # Add temporal consistency result in the list
            self.temporal_consistency.gather_one_by_one_inference(out)
            # Check if the buffer is filled with a full batch
            if self.temporal_consistency.is_buffer_complete():
                out = self.temporal_consistency.frames_inference.clone().to(self.device)
                # batch, 81, 144
            else:
                return 'buffer not full yet'
        # Apply opening and closing in binarized (0~1) image
        # batch, 81, 144
        out = self.opening(out)
        # batch, 1 81, 144
        out = self.closing(out)
        # batch, 1, 81, 144
        if tensors['cycle_name'] == 'training MM':
            return out
        # Count pixels "on" in each image
        out = self.sum_pixels_on(out)
        # batch, 1
        # Now we have the amount of pixels on, we will transform it linearly to have a range between 0 and 1
        out = self.change_scale_for_classification(out)
        # batch, 1
        # Apply a threshold to have the final classification
        out = self.classification_function(out)
        # batch, 1
        for i, o in enumerate(out):
            if o.item() < 0:
                out[i] += (-1. * o)
                print('negativo')
            if o.item() > 1:
                out[i] += (1. - o)
                print('maior que 1')
        for i in out:
            assert i.item() >= 0 and i.item() <= 1
        return out

    def get_trainable_values(self):
        return {
            'dissimilarity_module': {
                'weights_ref':
                self.dissimilarity_module.branches[0].weights_ref.cpu().detach().numpy(),
                'weights_tar':
                self.dissimilarity_module.branches[0].weights_tar.cpu().detach().numpy(),
                'bias_diff':
                self.dissimilarity_module.branches[0].bias_diff.cpu().detach().numpy(),
                'weights_channel':
                self.dissimilarity_module.branches[0].weights_channels.cpu().detach().numpy(),
                'combination_bias':
                self.dissimilarity_module.combination_bias.item()
            },
            'temporal_consistency': {
                'voting_window': self.temporal_consistency.voting_window
            },
            'opening': {
                'radius': self.opening.se_sigmoid.radius.item()
            },
            'closing': {
                'radius': self.closing.se_sigmoid.radius.item()
            },
            'classification_function': {
                'threshold': self.classification_function.threshold.item()
            },
        }

    def in_training(self):
        ret = {}
        # Módulo de diferença
        t = (self.dissimilarity_module.combination_bias.requires_grad +
             self.dissimilarity_module.branches[0].weights_ref.requires_grad +
             self.dissimilarity_module.branches[0].weights_tar.requires_grad +
             self.dissimilarity_module.branches[0].bias_diff.requires_grad +
             self.dissimilarity_module.branches[0].weights_channels.requires_grad) / 5
        ret['DM'] = f'{t} in training'
        # Opening
        t = (self.opening.pipeline.conv_dilate.radius.requires_grad +
             self.opening.pipeline.conv_erosion.radius.requires_grad +
             self.opening.se_sigmoid.radius.requires_grad) / 3
        ret['opening'] = f'{t} in training'
        # Closing
        t = (self.closing.pipeline.conv_dilate.radius.requires_grad +
             self.closing.pipeline.conv_erosion.radius.requires_grad +
             self.closing.se_sigmoid.radius.requires_grad) / 3
        ret['closing'] = f'{t} in training'
        # Classification threshold
        t = self.classification_function.threshold.requires_grad / 1
        ret['classification'] = f'{t} in training'
        return ret

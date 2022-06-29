from abc import ABC

import torch.nn as nn

from .modules.differential_morphology import (DifferentialClosing,
                                              DifferentialOpening,)
from .modules.dissimilarity import N_TensorsDifference
from .modules.dissimilarity_disteuc import DistEucModule
from .modules.modules_functions import (ChangeDynamicRange, LearnableSigmoid,
                                        SumAllPixels,)
from .modules.temporal_consistency import TemporalConsistency


class ABC_Pipeline(nn.Module, ABC):
    def __init__(self, net_params, ablation_cycles=None, ablation_name=None, device=None):
        super(ABC_Pipeline, self).__init__()
        self.device = device
        # DM
        if 'training DM' in ablation_cycles:
            if ablation_name == 'modification1':  # Dist euc
                self.dissimilarity_module = DistEucModule()
            else:
                distance_params = net_params['distance_module']
                self.dissimilarity_module = N_TensorsDifference(
                    n_pairs=distance_params['pairs'],
                    gamma_sigmoid=distance_params['gamma_bin_sigmoid'])
        # TCM
        if 'training TCM' in ablation_cycles:
            self.temporal_consistency = TemporalConsistency(
                max_frames=net_params['temporal_consistency_module']['max_frames'],
                type_voting=net_params['temporal_consistency_module']['type'])
        # MM
        if 'training MM' in ablation_cycles:
            morph_params = net_params['morphology_module']
            self.max_radius = morph_params['max_radius']
            self.opening = DifferentialOpening(
                se_gamma=morph_params['open']['se_gamma'],
                se_radius=morph_params['open']['radius'],
                kwargs_thresh_params=morph_params['open']['params_thresh'],
                epsilon=morph_params['open']['epsi'],
            )
            self.closing = DifferentialClosing(
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
        self.classification_function = LearnableSigmoid(threshold=classification_params['thresh'],
                                                        gamma=classification_params['gamma'])

    def get_trainable_values(self):
        if hasattr(self, 'dissimilarity_module'):
            if isinstance(self.dissimilarity_module, DistEucModule):
                _dissimilarity_module = {
                    'combination_bias': self.dissimilarity_module.combination_bias.item()
                }
            else:
                _dissimilarity_module = {
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
                }
        else:
            _dissimilarity_module = {}
        _temporal_consistency_module = {
            'voting_window': self.temporal_consistency.voting_window
        } if hasattr(self, 'temporal_consistency') else {}
        _opening = {
            'radius': self.opening.se_sigmoid.radius.item()
        } if hasattr(self, 'opening') else {}
        _closing = {
            'radius': self.closing.se_sigmoid.radius.item()
        } if hasattr(self, 'closing') else {}
        _classification = {
            'threshold': self.classification_function.threshold.item()
        } if hasattr(self, 'classification_function') else {}
        return {
            'dissimilarity_module': _dissimilarity_module,
            'temporal_consistency': _temporal_consistency_module,
            'opening': _opening,
            'closing': _closing,
            'classification_function': _classification
        }

    def in_training(self):
        ret = {}
        # Módulo de diferença
        if isinstance(self.dissimilarity_module, DistEucModule):
            t = self.dissimilarity_module.combination_bias.requires_grad
        else:
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

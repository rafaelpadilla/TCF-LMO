import datetime
import json
import os
import pickle
import shutil
import socket
import sys
import time

import click
import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import src.utils.utils_dataset as utils_dataset
import src.utils.utils_functions as utils_functions
import src.utils.utils_metrics as utils_metrics
import src.utils.utils_tensorboard as tb_utils
from src.lmdb_ds import LMDBDataset
from src.net_DM_MM_TCM_CM import Pipeline as model_DM_MM_TCM_CM
from src.net_DM_TCM_MM_CM import Pipeline as model_DM_TCM_MM_CM
from src.utils.VDAO_folds.Resnet50_reduced import Resnet50_Reduced


def print_info(text, log_path):
    utils_functions.log(log_path, text, option='a', print_out=True, new_line=True)


def print_validation_info(metrics_val, log_path):
    print_info(f'validation loss: {metrics_val["summary_validation"]["loss_validation"]}', log_path)
    print_info(f'validation accuracy: {(100*metrics_val["summary_validation"]["accuracy"]):.2f}%',
               log_path)
    print_info(f'validation DIS: {metrics_val["summary_validation"]["DIS_validation"]}', log_path)


def print_net_params(model, log_path):
    print_info('\nMain parameters of the network:', log_path)
    print_info(f'* opening radius: {model.opening.se_sigmoid.radius.item()}', log_path)
    print_info(f'* closing radius: {model.closing.se_sigmoid.radius.item()}', log_path)
    print_info(f'* temporal consistency neighbors: {model.temporal_consistency.voting_window}',
               log_path)
    print_info(f'* threshold classification: {model.classification_function.threshold.item()}\n',
               log_path)


def print_training_info(cycle_name, loss_epoch_train, start_time, log_path):
    print_info(
        f'training cycle {cycle_name}\tLoss: {loss_epoch_train}\tRunning time: {time.time() - start_time} s',
        log_path)


def unnormalize_add_bb(frame_norm, std_val, mean_val, bb=None):
    f = utils_functions.unnormalize(frame_norm.squeeze(), std_val, mean_val,
                                    one_channel=False).permute(1, 2, 0)
    f = (255 * f.numpy()).astype(np.uint8)
    if bb is not None:
        f = utils_functions.add_bb_into_image(f.copy(),
                                              bb.squeeze().numpy(),
                                              color=(0, 0, 255),
                                              thickness=2,
                                              label=None)
    return f


def show_frame(frame_norm, std_val, mean_val, bb):
    f = unnormalize_add_bb(frame_norm, std_val, mean_val, bb)
    Image.fromarray(f).show()


@click.command()
@click.option('--fold', default=-1, help='Fold number.', type=click.IntRange(1, 9, clamp=False))
@click.option('--batch_size', default=14, help='Batch size.', type=click.INT)
@click.option('--epochs', default=100, help='Number of epochs to train.', type=click.INT)
@click.option('--device', default=0, help='GPU device.', type=click.INT)
@click.option('--perform_validation/--no-perform_validation',
              default=True,
              help='If present, performs validation after every epoch.')
@click.option(
    '--run_once_without_training/--no-run_once_without_training',
    default=True,
    help=
    'If present, before training, performs validation on the first epoch, so the first assessment metrics and network parameters are stored.'
)
@click.option('--net',
              default='DM_MM_TCM_CM',
              help='Network structure.',
              type=click.Choice(['DM_MM_TCM_CM', 'DM_TCM_MM_CM'], case_sensitive=False))
@click.option('--name_experiment', default=None, help='name of the experiment.', required=False)
@click.option('--seed',
              default=123,
              help='random seed to achieve achieve reproducible results.',
              type=click.INT)
@click.option("--init_params_file",
              type=click.Path(exists=True),
              default='src/init_params_train.json',
              required=False)
@click.option("--continue_from", type=click.Path(exists=True), required=False)
@click.option("--tb_params_file",
              type=click.Path(exists=True),
              default='src/tb_params.json',
              required=False)
@click.option('--alignment',
              default='temporal',
              help='Type of alignment alignment.',
              type=click.Choice(['temporal', 'geometric'], case_sensitive=False))
def main(fold, epochs, batch_size, net, name_experiment, seed, init_params_file, tb_params_file,
         device, perform_validation, run_once_without_training, continue_from, alignment):

    # Read init params file
    init_params = json.load(open(init_params_file, 'r'))
    # Read tb params file
    tensorboard_params = json.load(open(tb_params_file, 'r'))

    if continue_from is not None:
        log_dir = os.path.split(continue_from)[0]
        run_once_without_training = False
    else:
        if name_experiment is None:
            name_experiment = f'training_fold_{fold}'
        log_dir = os.path.join('training_logs', name_experiment)
    log_path = os.path.join(log_dir, 'logging.txt')
    # Check if there is already a folder with the name of the experiment
    create_folder = True
    dir_exists = os.path.isdir(log_dir)
    # If it shouldnt continue from an existing experiment, check if the experiment ex
    if continue_from is None and dir_exists:
        create_folder = input(
            f'A directory with the name of the experiment ({log_dir}) already exist.\nDo you want to overwrite it? (y: yes /n: no) '
        ) == 'y'
        if create_folder:
            create_folder = input(
                'REALLY? ARE YOU SURE? ALL CONTENT WILL BE ERASED! (y: yes /n: no) ') == 'y'
            if create_folder:
                shutil.rmtree(log_dir)
    if continue_from is None and create_folder is False:
        print('Exiting...')
        sys.exit()

    # create tensorboard
    writer = SummaryWriter(log_dir=log_dir)
    # log parameters
    print_info(f'Folder {log_dir} created to save tensorboard logs.\n', log_path)
    print_info(f'Started at: {datetime.datetime.now()}', log_path)
    print_info(f'Hostname: {socket.gethostname()}', log_path)
    print_info(f'Experiment name: {name_experiment}\n', log_path)
    print_info(f'Fold: {fold}', log_path)
    print_info(f'Parameters:', log_path)
    print_info(f'--fold {fold}', log_path)
    print_info(f'--batch_size {batch_size}', log_path)
    print_info(f'--epochs {epochs}', log_path)
    print_info(f'--net {net}', log_path)
    print_info(f'--name_experiment {name_experiment}', log_path)
    print_info(f'--seed {seed}', log_path)
    print_info(f'--init_params_file {init_params_file}', log_path)
    print_info(f'--device {device}', log_path)
    print_info(f'--tb_params_file {tb_params_file}', log_path)
    print_info(f'--perform_validation {perform_validation}', log_path)
    print_info(f'--run_once_without_training {run_once_without_training}', log_path)
    print_info(f'--continue_from {continue_from}', log_path)
    print_info(f'--alignment {alignment}', log_path)
    print_info(f'\n', log_path)

    # Define cycles and batch sizes
    if net == 'DM_MM_TCM_CM':
        train_cycles = [
            {
                'cycle_name': 'training DM',
                'loss_func': nn.MSELoss(),
                'batch_size': batch_size,
                'load_mode': 'keyframe',
                'count_trained_batches': 0
            },
            {
                'cycle_name': 'training MM',
                'loss_func': nn.MSELoss(),
                'batch_size': batch_size,
                'load_mode': 'keyframe',
                'count_trained_batches': 0
            },
            {
                'cycle_name': 'training TCM',
                'loss_func': nn.MSELoss(),
                'batch_size': 1,  # 1 block
                'load_mode': 'block',
                'count_trained_batches': 0
            },
            {
                'cycle_name': 'training CM',
                'loss_func': nn.MSELoss(),
                'batch_size': 1,  # 1 block
                'load_mode': 'block',
                'count_trained_batches': 0
            }
        ]
    else:  # net == 'DM_TCM_MM_CM'
        train_cycles = [
            {
                'cycle_name': 'training DM',
                'loss_func': nn.MSELoss(),
                'batch_size': batch_size,
                'load_mode': 'keyframe',
                'count_trained_batches': 0
            },
            {
                'cycle_name': 'training TCM',
                'loss_func': nn.MSELoss(),
                'batch_size': 1,  # 1 block
                'load_mode': 'block',
                'count_trained_batches': 0
            },
            {
                'cycle_name': 'training MM',
                'loss_func': nn.MSELoss(),
                'batch_size': 1,  # 1 passa-se com blocos acumulados
                'load_mode': 'block',
                'count_trained_batches': 0
            },
            {
                'cycle_name': 'training CM',
                'loss_func': nn.MSELoss(),
                'batch_size': 1,  # passa-se com blocos acumulados
                'load_mode': 'block',
                'count_trained_batches': 0
            },
        ]
    # Cycle for validation or testing
    inference_cycle = {
        'cycle_name': 'inference',
        'load_mode': 'keyframe',
        'loss_func': nn.MSELoss(),
    }
    # Define device
    torch.cuda.set_device(device)
    device = torch.device(f'cuda:{device}') if torch.cuda.is_available() else torch.device('cpu')
    # Create CNN for feature extraction
    resnet = Resnet50_Reduced(device)
    resnet.freeze()

    init_params['scale_module'] = init_params['scale_module'][alignment]
    # Create network
    if net == 'DM_MM_TCM_CM':
        model = model_DM_MM_TCM_CM(init_params, device).to(device)
    else:  # DM_TCM_MM_CM
        model = model_DM_TCM_MM_CM(init_params, device).to(device)

    hooks_dict = utils_functions.register_hooks(model)

    # Continue from previous training
    if continue_from:
        model = torch.load(continue_from, map_location=device)
        model.device = device
        print_info(f'Continuing training from {continue_from}\n', log_path)

    ###################################################
    # Training parameters
    ###################################################
    train_params = {'num_epochs': epochs, 'lr': 1e-8}

    optimizer = optim.Adam([
        {
            'params': model.dissimilarity_module.branches[0].weights_ref,
            'lr': 1e-2
        },
        {
            'params': model.dissimilarity_module.branches[0].weights_tar,
            'lr': 1e-2
        },
        {
            'params': model.dissimilarity_module.branches[0].bias_diff,
            'lr': 2e-4
        },
        {
            'params': model.dissimilarity_module.branches[0].weights_channels,
            'lr': 2e-2
        },
        {
            'params': model.dissimilarity_module.combination_bias,
            'lr': 2e-3
        },
        {
            'params': model.opening.parameters(),
            'lr': 1e-4
        },
        {
            'params': model.closing.parameters(),
            'lr': 13e-3
        },
        {
            'params': model.classification_function.parameters(),
            'lr': 1e-4
        },
    ],
                           lr=train_params['lr'])

    # As frames in the LMDB are normalized, lets define the denormalizations
    normalize_transform = transforms.Normalize(mean=resnet.MEAN_IMAGENET, std=resnet.STD_IMAGENET)
    to_tensor_transform = transforms.ToTensor()
    transformations = transforms.Compose([to_tensor_transform, normalize_transform])

    ###################################################
    # Datasets and dataloaders
    ###################################################
    # Training dataset
    print_info(f'Datasets info:', log_path)
    loader_params_train = {'shuffle': True, 'num_workers': 0, 'worker_init_fn': seed}
    dataset_train = LMDBDataset(fold_number=fold,
                                type_dataset='train',
                                transformations=transformations,
                                balance=True,
                                load_mode='block',
                                alignment=alignment)
    total_pos = len([b for b in dataset_train.keys_ds if b['class_keyframe'] is True])
    total_neg = len([b for b in dataset_train.keys_ds if b['class_keyframe'] is False])

    print_info(f'Training dataset (fold {fold}) loaded with {len(dataset_train)} samples:',
               log_path)
    print_info(f'Positive samples: {total_pos}', log_path)
    print_info(f'Negative samples: {total_neg}', log_path)
    print_info(str(dataset_train.get_objects()) + '\n', log_path)
    # Validation dataset
    loader_params_val = {'shuffle': False, 'num_workers': 0, 'worker_init_fn': seed}
    dataset_validation = LMDBDataset(fold_number=fold,
                                     type_dataset='validation',
                                     transformations=transformations,
                                     balance=False,
                                     load_mode='keyframe',
                                     max_samples=None,
                                     alignment=alignment)
    datasets_validation = utils_dataset.split_data_set_into_videos_lmdb(dataset_validation)
    total_pos = len([b for b in dataset_validation.keys_ds if b['class_keyframe'] is True])
    total_neg = len([b for b in dataset_validation.keys_ds if b['class_keyframe'] is False])
    print_info(f'Validation dataset (fold {fold}) loaded with {len(dataset_validation)} samples:',
               log_path)
    print_info(f'Positive samples: {total_pos}', log_path)
    print_info(f'Negative samples: {total_neg}', log_path)
    print_info(str(dataset_validation.get_objects()) + '\n', log_path)

    def prepare_model(cycle, len_data_loader_train):
        '''freeze/unfreeze modules'''

        # Pass to TCM the amount of samples and the batch size. This is needed to accumulate the samples
        model.temporal_consistency.set_batch_info(total_samples=len_data_loader_train,
                                                  samples_to_accumulate=batch_size)
        # Freeze all modules
        model.dissimilarity_module.freeze()
        model.opening.freeze()
        model.closing.freeze()
        model.classification_function.freeze()
        # Unfreeze the needed modules, depending on the training cycle
        if cycle['cycle_name'] == 'training DM':
            model.dissimilarity_module.unfreeze()
        elif cycle['cycle_name'] == 'training TCM':
            pass
        elif cycle['cycle_name'] == 'training MM':
            model.opening.unfreeze()
            model.closing.unfreeze()
        elif cycle['cycle_name'] == 'training CM':
            model.classification_function.unfreeze()
        elif cycle['cycle_name'] == 'inference':
            # Do nothing, once all modules are frozen
            pass

    def prepare_samples(cycle, ref_frames, tar_frames, labels_classes, bbs):
        ''' samples arrive in the format (batch, samples, channel, h, w). Depending on the cycle, arrange the samples and dimensions'''
        if cycle['cycle_name'] == 'training DM':
            # if 'training DM' -> load_mode is 'keyframe' -> samples=1 :. (batch, 1, c, h, w)
            optimizer.zero_grad()
        elif cycle['cycle_name'] == 'training TCM':
            # if 'training TCM' -> load_mode is 'block' -> samples=15 -> batch=1 :. (1, 15, c, h, w)
            # Thus, if 'training TCM', it is needed to switch samples <-> batch, so it becomes (15, 1, c, h, w)
            # A squeeze(1) is needed so it becomes (15, c, h, w)
            ref_frames = ref_frames.permute(1, 0, 2, 3, 4).squeeze(1)
            tar_frames = tar_frames.permute(1, 0, 2, 3, 4).squeeze(1)
            bbs = bbs.permute(1, 0, 2).squeeze(1)
            # labels_classes = # Do nothing, because it is a list
            model.temporal_consistency.start_new_train_batch()
        elif cycle['cycle_name'] == 'training MM':
            optimizer.zero_grad()
            if net == 'DM_TCM_MM_CM':
                # squeeze(1) so it becomes 15, c, h, w
                ref_frames = ref_frames.permute(1, 0, 2, 3, 4).squeeze(1)
                tar_frames = tar_frames.permute(1, 0, 2, 3, 4).squeeze(1)
                # get central id of the block
                middle_id = model.temporal_consistency.max_frames // 2
                bbs = bbs.permute(1, 0, 2)[middle_id]
                labels_classes = labels_classes[middle_id]
        elif cycle['cycle_name'] == 'training CM':
            optimizer.zero_grad()
            # squeeze(1) so it becomes (15, c, h, w)
            ref_frames = ref_frames.permute(1, 0, 2, 3, 4).squeeze(1)
            tar_frames = tar_frames.permute(1, 0, 2, 3, 4).squeeze(1)
            # gets the central id of the block
            middle_id = model.temporal_consistency.max_frames // 2
            bbs = bbs.permute(1, 0, 2)[middle_id]
            labels_classes = labels_classes[middle_id]
        elif cycle['cycle_name'] == 'inference':
            pass

        return ref_frames, tar_frames, labels_classes, bbs

    def tb_log_image_strips(data_dict, counter):
        data_dict['labels_bool'] = [i.item() == 1 for i in data_dict['gt_labels']]
        outputs = {
            'model_output': [o.item() for o in data_dict['outputs']],
            'output_bool': [o.item() > .5 for o in data_dict['outputs']]
        }
        # log into tensorboard
        ref_img = (tb_utils.unnormalize(torch.stack(data_dict['ref_frame']),
                                        resnet.STD_IMAGENET,
                                        resnet.MEAN_IMAGENET,
                                        one_channel=False) * 255).to(torch.uint8)
        tar_img = (tb_utils.unnormalize(torch.stack(data_dict['tar_frame']),
                                        resnet.STD_IMAGENET,
                                        resnet.MEAN_IMAGENET,
                                        one_channel=False) * 255).to(torch.uint8)
        params_dict = {
            'desired_output_shape': ref_img.shape,
            'ref_img': ref_img,
            'tar_img': tar_img,
            'dissimilarity_output': torch.stack(data_dict['dissimilarity_output']),
            'temporal_consistency_output': data_dict['temporal_consistency_output'],
            'opening_output': data_dict['opening_output'],
            'closing_output': data_dict['closing_output'],
            'gt_classes': data_dict['labels_bool'],
            'preds_classes': outputs['output_bool'],
            'rad_open': model.opening.se_sigmoid.radius.item(),
            'thresh_open': model.opening.thresh_by_volume_erosion.volume,
            'rad_close': model.closing.se_sigmoid.radius.item(),
            'thresh_close': model.closing.thresh_by_volume_erosion.volume,
            'outputs_model': outputs['model_output']
        }
        img_results_strap = tb_utils.get_strips_intermediate_images(**params_dict)
        writer.add_image('final_strip', img_results_strap, global_step=counter)
        writer.close()

    def train(cycle, not_learning=False):
        # Set load_mode ('block' or 'keyframe') and batch size according to the training cycle
        dataset_train.load_mode = cycle['load_mode']
        data_loader_train = DataLoader(dataset_train,
                                       **loader_params_train,
                                       batch_size=cycle['batch_size'])

        # prepare model (freezing modules that are not trained)
        prepare_model(cycle, len(data_loader_train))

        frames_to_save = {
            'ref_frame': [],
            'tar_frame': [],
            'gt_labels': [],
            'dissimilarity_output': [],
            'temporal_consistency_output': None,
            'opening_output': [],
            'closing_output': [],
            'outputs': None
        }

        # dictionary to be updated with the best temporal windows every epoch
        dict_temporal_consistency_results = {}
        losses = []

        for batch, (ref_frames, tar_frames, labels_classes, bbs) in enumerate(data_loader_train):
            # prepare samples (permute channels, zero grads, etc)
            (ref_frames, tar_frames, labels_classes,
             bbs) = prepare_samples(cycle, ref_frames, tar_frames, labels_classes, bbs)

            middle_id = model.temporal_consistency.max_frames // 2

            # if in the last cycle, check if it is time to save images on the tensorboard
            save_images_tb = tensorboard_params['training']['intermediate_images']['enabled'].lower(
            ) == 'true' and cycle['cycle_name'] == 'training CM' and cycle[
                'count_trained_batches'] % tensorboard_params['training']['intermediate_images'][
                    'period'] == 0
            if save_images_tb:
                frames_to_save['ref_frame'].append(ref_frames[middle_id].cpu())
                frames_to_save['tar_frame'].append(tar_frames[middle_id].cpu())
                frames_to_save['gt_labels'].append(labels_classes.cpu())

            gts = {
                'labels': labels_classes,
                'bounding_boxes': bbs,
                # 'shape': tar_frames.squeeze().shape
            }
            # features from the frames
            feat_ref = resnet(ref_frames.to(device))
            feat_tar = resnet(tar_frames.to(device))

            # if there is only 1 sample in the batch len(feat_ref.shape) == 3
            if feat_ref.dim() == 3:
                feat_ref = feat_ref.unsqueeze(0)
            if feat_tar.dim() == 3:
                feat_tar = feat_tar.unsqueeze(0)

            # pass samples by the network
            outputs = model({
                'feat_ref': feat_ref,
                'feat_tar': feat_tar,
                'cycle_name': cycle['cycle_name']
            })

            # if it is time to save images of this batch on the tensorboard
            if save_images_tb:
                if net == 'DM_MM_TCM_CM':
                    frames_to_save['dissimilarity_output'].append(
                        (hooks_dict['hook_opening'].input[0][middle_id] * 255).to(
                            torch.uint8).cpu())
                    frames_to_save['opening_output'].append(
                        (hooks_dict['hook_opening'].output[middle_id] * 255).to(
                            torch.uint8).cpu().squeeze())
                    frames_to_save['closing_output'].append(
                        (hooks_dict['hook_closing'].output[middle_id] * 255).to(
                            torch.uint8).cpu().squeeze())
                else:  # net == 'DM_TCM_MM_CM':
                    frames_to_save['dissimilarity_output'].append(
                        (hooks_dict['hook_dissimilarity'].output[middle_id] * 255).to(
                            torch.uint8).cpu())

            # calculates the loss according to the cycle
            if cycle['cycle_name'] == 'training DM':
                # compute normalized MCC (between 0 and 1)
                norm_mcc = utils_functions.calculate_norm_mcc(output=outputs,
                                                              gt=gts,
                                                              alignment=alignment,
                                                              device=device)
                # when optimizing normalized mcc, the expected output is 1
                labels = torch.ones_like(norm_mcc).to(device)
                loss = cycle['loss_func'](norm_mcc, labels)
                losses.append(loss.item())
                if not not_learning:
                    loss.backward()
                    optimizer.step()
            elif cycle['cycle_name'] == 'training TCM':
                # get the central frame, which is the representative frame of the block
                gts['middle_id'] = model.temporal_consistency.max_frames // 2
                # compute normalized MCC (between 0 and 1)
                # outputs => output of the network training with TCM is a dict containing the amount of pixels "on" in each window size 1,3,5,7,9,11,13,15
                norm_mcc_dict = utils_functions.calculate_best_window_temporal_consistency(
                    outputs, gts, alignment=alignment, device=device)
                # when optimizing normalized mcc, the expected output is 1
                labels = torch.ones([1]).to(device).unsqueeze(0)
                # compute mcc loss for each voting window
                loss = {
                    window: cycle['loss_func'](norm_mcc, labels).item()
                    for window, norm_mcc in norm_mcc_dict.items()
                }
                losses.append(loss)
            elif cycle['cycle_name'] == 'training MM':

                if net == 'DM_TCM_MM_CM':
                    # in every training loop, a batch with samples enter the net and, while passing by the TCM, they are transformed into 1 sample. This sample is stored in models.temporal_consistency.frames_inference until it reaches 14 samples. Thats why we need to store the gt label in the list model.temporal_consistency.gt_labels_inference
                    model.temporal_consistency.gather_gt_label_inference(gts['labels'] * 1.)
                    # We also store each bounding box in the list model.temporal_consistency.gt_bbs_inference
                    model.temporal_consistency.gather_gt_bb_inference(
                        gts['bounding_boxes'].squeeze())
                    if outputs == 'buffer not full yet':
                        continue
                    # MM is optimized with MCC of the output image (white blob) and image with bb
                    # computes normalized MCC, with values between 0 and 1
                    gts = {
                        'bounding_boxes': model.temporal_consistency.gt_bbs_inference,
                        # 'shape': tar_frames.squeeze().shape,
                        'labels': model.temporal_consistency.gt_labels_inference
                    }
                # compute normalized MCC (between 0 and 1)
                norm_mcc = utils_functions.calculate_norm_mcc(output=outputs,
                                                              gt=gts,
                                                              alignment=alignment,
                                                              device=device)
                # normalized mcc is expected to be 1
                labels = torch.ones_like(norm_mcc).to(device)
                loss = cycle['loss_func'](norm_mcc, labels)
                losses.append(loss.item())
                if not not_learning:
                    loss.backward()
                    optimizer.step()
                # Clean temporal buffer
                if net == 'DM_TCM_MM_CM':
                    model.temporal_consistency.batch_sizes.pop(0)
                    model.temporal_consistency.clean_buffer()
            elif cycle['cycle_name'] == 'training CM':
                # Aggregates in the gts list, the gt label (multiply to 1 to transform the bool into 0 or 1)
                model.temporal_consistency.gather_gt_label_inference(gts['labels'] * 1.)
                if outputs == 'buffer not full yet':
                    continue
                # compute loss (MSE of the percentage of the image with pixels "on")
                if net == 'DM_MM_TCM_CM':
                    gts_labels = torch.tensor(
                        model.temporal_consistency.gt_labels_inference).to(device)
                else:  # net == 'DM_TCM_MM_CM'
                    gts_labels = torch.tensor(
                        model.temporal_consistency.gt_labels_inference).unsqueeze(1).to(device)
                loss = cycle['loss_func'](outputs, gts_labels)
                losses.append(loss.item())
                if not not_learning:
                    loss.backward()
                    optimizer.step()
                # Clear the buffer of the temporal voting
                model.temporal_consistency.batch_sizes.pop(0)
                model.temporal_consistency.clean_buffer()
            if save_images_tb:
                if net == 'DM_MM_TCM_CM':
                    frames_to_save['temporal_consistency_output'] = (
                        hooks_dict['hook_sum_pixels_on'].input[0].squeeze() * 255).to(
                            torch.uint8).cpu()
                    frames_to_save['outputs'] = outputs
                    # Transform lists into tensors
                    frames_to_save['closing_output'] = torch.stack(frames_to_save['closing_output'])
                    frames_to_save['opening_output'] = torch.stack(frames_to_save['opening_output'])
                else:  # net == 'DM_TCM_MM_CM':
                    frames_to_save['temporal_consistency_output'] = (
                        hooks_dict['hook_opening'].input[0] * 255).to(torch.uint8).cpu()
                    frames_to_save['opening_output'] = (hooks_dict['hook_opening'].output * 255).to(
                        torch.uint8).cpu()
                    frames_to_save['closing_output'] = (hooks_dict['hook_closing'].output * 255).to(
                        torch.uint8).cpu()
                    frames_to_save['outputs'] = outputs
                tb_log_image_strips(frames_to_save, cycle['count_trained_batches'])
                frames_to_save = {
                    'ref_frame': [],
                    'tar_frame': [],
                    'gt_labels': [],
                    'dissimilarity_output': [],
                    'temporal_consistency_output': None,
                    'opening_output': [],
                    'closing_output': [],
                    'outputs': None
                }

            cycle['count_trained_batches'] += 1

        ################################################################################################
        # Finished the cycle
        ################################################################################################
        if cycle['cycle_name'] == 'training DM':
            mean_loss = np.mean(losses)
        elif cycle['cycle_name'] == 'training TCM':
            # for each batch, sum all results of the voting window
            d = {}
            for l in losses:
                for k, v in l.items():
                    d.setdefault(k, []).append(v)
            d = {k: sum(v) for k, v in d.items()}
            for k in d.keys():
                if k not in dict_temporal_consistency_results:
                    dict_temporal_consistency_results[k] = 0
            # Add into dict_temporal_consistency_results the sum of the results obtained in this epoch
            for window, soma in d.items():
                dict_temporal_consistency_results[window] += soma
            # Set into the model, the window with the lowest loss
            model.temporal_consistency.voting_window = min(
                dict_temporal_consistency_results, key=dict_temporal_consistency_results.get)
            # compute the mean loss
            mean_loss = d[model.temporal_consistency.voting_window] / \
                len(losses)
        elif cycle['cycle_name'] == 'training MM':
            mean_loss = np.mean(losses)
        elif cycle['cycle_name'] == 'training CM':
            mean_loss = np.mean(losses)
        return mean_loss

    def validate(cycle, quiet=True):
        metrics_all_videos = {'videos': {}, 'summary_validation': None}
        pred_labels, gt_labels = [], []

        # In the validation phase, it is needed to load one video at a time
        for id_vid_val, dataset_val in enumerate(datasets_validation):
            # makes sure that there is only 1 video that is being loaded
            assert len(set([k['video_name'] for k in dataset_val.keys_ds])) == 1
            vid_basename = dataset_val.keys_ds[0]['video_name']

            if not quiet:
                print(f'Evaluating video {vid_basename} ({len(dataset_val)} frames)')

            data_loader_validate = DataLoader(dataset_val,
                                              **loader_params_val,
                                              batch_size=model.temporal_consistency.voting_window)
            # prepare model (freezing modules that are not trained)
            prepare_model(cycle, len(data_loader_validate))

            metrics_vid = {
                'pred_labels': [],
                'gt_labels': [],
                'DIS': None,
                'accuracy': None,
                'mean_loss': None
            }
            losses_vid = []
            buffer_frames = {}
            count_samples = 0
            init_frame, central_frame, end_frame = 0, 0, 0
            voting_window = model.temporal_consistency.voting_window

            for batch, (ref_frames, tar_frames, labels_classes,
                        bbs) in enumerate(data_loader_validate):
                # prepare samples (permute channels, zero grads, etc)
                (ref_frames, tar_frames, labels_classes,
                 _) = prepare_samples(cycle, ref_frames, tar_frames, labels_classes, bbs)

                # features from the frames
                feat_ref = resnet(ref_frames.to(device))
                feat_tar = resnet(tar_frames.to(device))

                # if there is only 1 sample in the batch len(feat_ref.shape) == 3
                if len(feat_ref.shape) == 3:
                    feat_ref = feat_ref.unsqueeze(0)
                    feat_tar = feat_tar.unsqueeze(0)
                    labels_classes = labels_classes.unsqueeze(0)

                # aaa = (utils_functions.unnormalize(tar_frames, resnet.STD_IMAGENET, resnet.MEAN_IMAGENET).permute(0,2,3,1).numpy().squeeze()*255).astype(np.uint8)
                # for i, img in enumerate(aaa):
                #     Image.fromarray(img).save(f'{i}_tar.png')

                samples_batch = len(feat_ref)
                for i in range(samples_batch):
                    buffer_frames[count_samples] = {}
                    buffer_frames[count_samples]['feat_ref'] = feat_ref[i]
                    buffer_frames[count_samples]['feat_tar'] = feat_tar[i]
                    buffer_frames[count_samples]['class'] = labels_classes[i]
                    count_samples += 1

                init_frame = max(central_frame - voting_window // 2, 0)
                end_frame = min(central_frame + voting_window // 2, len(dataset_val))
                # clean the buffer => remove frames out of the voting window
                ids_to_remove = [i for i in buffer_frames if i < init_frame]
                for i in ids_to_remove:
                    del buffer_frames[i]

                while init_frame in buffer_frames and end_frame in buffer_frames and central_frame < len(
                        dataset_val):
                    # Sets the dictionary with the data to be passed to the network (between init_frame and end_frame)
                    data = {
                        'feat_ref': [],
                        'feat_tar': [],
                        'class': [],
                        'bb': [],
                        'frame_ids': [],
                        'central_frame': central_frame
                    }
                    for i in range(init_frame, end_frame + 1, 1):
                        {data[k].append(v) for k, v in buffer_frames[i].items()}
                        data['frame_ids'].append(i)
                    position_central_frame = data['frame_ids'].index(central_frame)

                    data['feat_ref'] = torch.stack(data['feat_ref'])
                    data['feat_tar'] = torch.stack(data['feat_tar'])

                    # Pass data through the network
                    outputs = model.inference_validation_test(data)

                    label_gt = (data['class'][position_central_frame] * 1.).to(device)
                    loss = cycle['loss_func'](outputs.squeeze(), label_gt)
                    losses_vid.append(loss.item())

                    central_frame += 1
                    init_frame = max(central_frame - voting_window // 2, 0)
                    end_frame = min(central_frame + voting_window // 2, len(dataset_val))

                    if end_frame >= len(dataset_val):
                        end_frame = len(dataset_val) - 1

                    # compute the metrics
                    class_out = (outputs > .5).item()
                    metrics_vid['gt_labels'].append((label_gt.item() == 1) * 1)
                    metrics_vid['pred_labels'].append(class_out * 1)

            # finished to validate one video
            # compute the metrics
            metrics_vid['mean_loss'] = np.mean(losses_vid)
            metrics_vid['DIS'] = utils_metrics.calculate_DIS(metrics_vid['pred_labels'],
                                                             metrics_vid['gt_labels'])
            metrics_vid['accuracy'] = utils_metrics.calculate_accuracy(
                metrics_vid['pred_labels'], metrics_vid['gt_labels'])
            # accumulate lists with predictions and groundtruths to be uses d in the final DIS
            pred_labels += metrics_vid['pred_labels']
            gt_labels += metrics_vid['gt_labels']
            # save matrics of the video
            metrics_all_videos['videos'][vid_basename] = metrics_vid

        # computes general metrics considering all videos
        metrics_all_videos['summary_validation'] = {
            'loss_validation':
            np.mean([met['mean_loss'] for vid, met in metrics_all_videos['videos'].items()]),
            'DIS_validation':
            utils_metrics.calculate_DIS(pred_labels, gt_labels),
            'accuracy':
            utils_metrics.calculate_accuracy(pred_labels, gt_labels)
        }
        return metrics_all_videos

    log_data = {'training_loss': {}, 'training_variables': {}, 'validation_metrics': {}}

    if continue_from is not None:
        init_epoch = int(
            os.path.basename(continue_from).replace('model_epoch_', '').replace('.pth', '')) + 1
    else:
        init_epoch = 0

    for epoch in range(init_epoch, epochs):
        print_info('*' * 100, log_path)
        # If first epoch requires no learning
        not_learning = run_once_without_training and epoch == 0
        if not_learning:
            print_info(f'\nEpoch {epoch+1}:{epochs} \t NOT LEARNING', log_path)
        else:
            print_info(f'\nEpoch {epoch+1}:{epochs}', log_path)
        # initiate the dictionary
        log_data['training_loss'][epoch] = {}
        # Training
        for train_cycle in train_cycles:
            start = time.time()
            cycle_name = train_cycle['cycle_name'].replace('training ', '')
            loss_epoch_train = train(train_cycle, not_learning=not_learning)
            print_training_info(cycle_name, loss_epoch_train, start, log_path)
            # register into the log (pickle file) the module loss
            log_data['training_loss'][epoch][train_cycle["cycle_name"]] = loss_epoch_train
            # save training loss into the tensorboard
            if epoch % tensorboard_params['training']['loss']['period'] == 0:
                writer.add_scalars("loss", {f'train {cycle_name}': loss_epoch_train}, epoch)
                writer.close()

        # after all cycles, store into log (pickle file) all parameters of the network
        log_data['training_variables'][epoch] = model.get_trainable_values()

        # after all cycles, save the learned parameters
        if epoch % tensorboard_params['training']['parameters_evolution']['period'] == 0:
            tb_utils.save_trainable_values(writer,
                                           model,
                                           prefix_var='epoch_',
                                           step=epoch,
                                           histogram=True)
        # after all cycles, save the model
        if epoch % tensorboard_params['training']['save_model']['period'] == 0:
            path_save_model = os.path.join(log_dir, f'model_epoch_{epoch}.pth')
            torch.save(model, path_save_model)
            print_info(f'\nModel saved (model_epoch_{epoch}.pth)', log_path)

        # Validation
        if perform_validation:
            print_info('\nEvaluating validation set', log_path)
            metrics_val = validate(inference_cycle)
            print_validation_info(metrics_val, log_path)

            log_data['validation_metrics'][epoch] = metrics_val
            # save valitation into tensorboard
            if epoch % tensorboard_params['validation']['loss']['period'] == 0:
                writer.add_scalars(
                    "loss", {'validation': metrics_val["summary_validation"]["loss_validation"]},
                    epoch)
                writer.add_scalars("accuracy_val",
                                   {'validation': metrics_val["summary_validation"]["accuracy"]},
                                   epoch)
                writer.add_scalars(
                    "DIS_val", {'validation': metrics_val["summary_validation"]["DIS_validation"]},
                    epoch)
                writer.close()
        # Print the parameters
        print_net_params(model, log_path)
        # Save pickle file
        filename = 'results'
        if continue_from is not None:  # if continued from a previously trained model
            files = utils_functions.get_files_recursively(log_dir, 'results*.pickle')
            if len(files) != 0:
                filename = f'results_continuation_{len(files)}'
        pickle_fp_to_save = os.path.join(f'{log_dir}', f'{filename}.pickle')
        pickle.dump(log_data, open(pickle_fp_to_save, 'wb'))
        print_info(f'\nPickles with results saved successfully ({pickle_fp_to_save})', log_path)


if __name__ == "__main__":
    # device = torch.device(f'cpu')
    # for fold in range(1, 10):
    #     try:
    #         if fold in [2, 3, 7, 8, 9]:
    #             model_path = f'/home/rafael.padilla/thesis/differentiable-anomaly-detection-pipeline/training_logs/temporal_alignment_fold_{fold}_reforco/model_epoch_0.pth'
    #         else:
    #             model_path = f'/home/rafael.padilla/thesis/differentiable-anomaly-detection-pipeline/training_logs/temporal_alignment_fold_{fold}/model_epoch_0.pth'
    #         model = torch.load(model_path, map_location=device)
    #         print(f'deu certo fold {fold}')
    #     except:
    #         print(f'deu merda fold {fold}')
    main()

# main(
#     fold=2,
#     device=0,
#     net='DM_MM_TCM_CM',
#     alignment='temporal',
#     name_experiment='temporal_alignment_fold_2_reforco',
#     continue_from='/training_logs/temporal_alignment_fold_2_reforco/model_epoch_82.pth',
#     batch_size=14,
#     epochs=100,
#     perform_validation=True,
#     run_once_without_training=False,
#     seed=123,
#     init_params_file='src/init_params_train.json',
#     tb_params_file='src/tb_params.json',
# )

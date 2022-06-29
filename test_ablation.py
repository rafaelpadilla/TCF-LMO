import datetime
import os
import pickle
from pathlib import Path

import click
import imageio
import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from jinja2 import ModuleLoader
from torch.utils.data import DataLoader

import src.utils.utils_dataset as utils_dataset
import src.utils.utils_functions as utils_functions
import src.utils.utils_metrics as utils_metrics
import src.utils.utils_tensorboard as tb_utils
from src.lmdb_ds import LMDBDataset
from src.utils.VDAO_folds.Resnet50_reduced import Resnet50_Reduced


def create_image_strips(data_dict, std_val, mean_val):
    # Define background of the non-image areas
    background_color = 'gray'
    ref_img = (tb_utils.unnormalize(
        data_dict['ref_frame'].unsqueeze(0), std_val, mean_val, one_channel=False) * 255).to(
            torch.uint8)
    tar_img = (tb_utils.unnormalize(
        data_dict['tar_frame'].unsqueeze(0), std_val, mean_val, one_channel=False) * 255).to(
            torch.uint8)
    # Create an image with reference and target frames side by side
    inputs = torch.cat((ref_img, tar_img), axis=0)
    color = 'red' if data_dict["gt_label"] else 'dark_green'
    ref_tar_strip = tb_utils.create_image_with_results(inputs,
                                                       ['ref', f'tar {data_dict["frame_id"]}'],
                                                       [color, color],
                                                       background=background_color,
                                                       scale_factor=1,
                                                       text_area_height=75,
                                                       font_size=15,
                                                       add_border=True)
    ref_tar_strip = np.moveaxis(ref_tar_strip, 0, -1)
    # Create an image with results DM an TCM side by side
    frames_to_add = []
    if 'DM' in data_dict: frames_to_add.append(data_dict['DM'].unsqueeze(0))
    if 'opening_output' in data_dict: frames_to_add.append(data_dict['opening_output'].unsqueeze(0))
    if 'closing_output' in data_dict: frames_to_add.append(data_dict['closing_output'].unsqueeze(0))
    if 'TCM' in data_dict: frames_to_add.append(data_dict['TCM'].unsqueeze(0))
    inputs = torch.cat(frames_to_add, axis=0)
    inputs = torch.cat(3 * [inputs.unsqueeze(0)])
    inputs = inputs.permute(1, 0, 2, 3)
    color = 'red' if data_dict["class_output"] else 'dark_green'

    images_texts = []
    if 'DM' in data_dict: images_texts.append('DM')
    if 'opening_output' in data_dict: images_texts.append(f'open {data_dict["rad_open"]:.2f}')
    if 'closing_output' in data_dict: images_texts.append(f'close {data_dict["rad_close"]:.2f}')
    if 'TCM' in data_dict: images_texts.append('TCM')
    colors_texts = ['black' for i in range(len(images_texts) - 1)]
    colors_texts = colors_texts + [color]
    results_strip = tb_utils.create_image_with_results(inputs,
                                                       images_texts,
                                                       colors_texts,
                                                       background=background_color,
                                                       scale_factor=1,
                                                       text_area_height=75,
                                                       font_size=15,
                                                       add_border=True)
    results_strip = np.moveaxis(results_strip, 0, -1)
    # Gathers in a single image all frames target and reference and the results
    H, W, C = ref_tar_strip.shape
    h, w, _ = results_strip.shape
    new_image = np.ones((h + H, W, C)).astype(np.uint8)
    for channel in range(C):
        new_image[:, :, channel] *= tb_utils.COLORS[background_color][channel]
    new_image[0:H, 0:W, :] = ref_tar_strip
    begin = (W - w) // 2
    new_image[H:, begin:begin + w, :] = results_strip
    return new_image


def print_info(text, log_path, init_block=False, end_block=False, sep='#'):
    if init_block:
        utils_functions.log(log_path, sep * 120, option='a', print_out=True, new_line=True)
    utils_functions.log(log_path, text, option='a', print_out=True, new_line=True)
    if end_block:
        utils_functions.log(log_path, sep * 120, option='a', print_out=True, new_line=True)


def get_frames_to_save(hooks_dict, modules, ablation, central_frame=0):

    frames_to_save = {}
    if ablation == 'modification1':
        if 'DM' in modules:
            frames_to_save['DM'] = (hooks_dict['hook_dissimilarity'].output * 255).to(
                torch.uint8).cpu()[central_frame]
        if 'TCM' in modules:
            frames_to_save['TCM'] = (hooks_dict['hook_sum_pixels_on'].input[0].squeeze() * 255).to(
                torch.uint8).cpu()
        if 'opening' in modules:
            frames_to_save['opening_output'] = (hooks_dict['hook_opening'].output[0].squeeze() *
                                                255).to(torch.uint8).cpu()
        if 'closing' in modules:
            frames_to_save['closing_output'] = (hooks_dict['hook_closing'].output[0].squeeze() *
                                                255).to(torch.uint8).cpu()
    elif ablation == 'modification2':
        if 'DM' in modules:
            frames_to_save['DM'] = (hooks_dict['hook_dissimilarity'].output * 255).to(
                torch.uint8).cpu()[central_frame]
        # generate frames to be included in the video
        if 'TCM' in modules:
            frames_to_save['TCM'] = (hooks_dict['hook_opening'].input[0].squeeze() * 255).to(
                torch.uint8).cpu()
        if 'opening' in modules:
            frames_to_save['opening_output'] = (hooks_dict['hook_opening'].output[0].squeeze() *
                                                255).to(torch.uint8).cpu()
        if 'closing' in modules:
            frames_to_save['closing_output'] = (hooks_dict['hook_closing'].output[0].squeeze() *
                                                255).to(torch.uint8).cpu()
    elif ablation == 'modification3':
        if 'DM' in modules:
            frames_to_save['DM'] = (hooks_dict['hook_dissimilarity'].output * 255).to(
                torch.uint8).cpu()[central_frame]
        if 'TCM' in modules:
            frames_to_save['TCM'] = (hooks_dict['hook_sum_pixels_on'].input[0].squeeze() * 255).to(
                torch.uint8).cpu()
        if 'opening' in modules:
            frames_to_save['opening_output'] = (hooks_dict['hook_opening'].output[0].squeeze() *
                                                255).to(torch.uint8).cpu()
        if 'closing' in modules:
            frames_to_save['closing_output'] = (hooks_dict['hook_closing'].output[0].squeeze() *
                                                255).to(torch.uint8).cpu()

    elif ablation == 'modification4':
        if 'DM' in modules:
            frames_to_save['DM'] = (hooks_dict['hook_dissimilarity'].output * 255).to(
                torch.uint8).cpu()

    return frames_to_save


def evaluate_model(model_path,
                   fold,
                   ablation,
                   device,
                   seed,
                   log_path,
                   alignment,
                   quiet=True,
                   dir_save=None,
                   save_videos=False,
                   save_frames=False,
                   quality=None,
                   fps=None):
    metrics_all_videos = {}

    # Load resnet
    resnet = Resnet50_Reduced(device)
    resnet.freeze()

    # As frames in the LMDB are normalized, lets define the normalization transformation
    normalize_transform = transforms.Normalize(mean=resnet.MEAN_IMAGENET, std=resnet.STD_IMAGENET)
    to_tensor_transform = transforms.ToTensor()
    transformations = transforms.Compose([to_tensor_transform, normalize_transform])

    # Load testing dataset
    ds = LMDBDataset(fold_number=fold,
                     type_dataset='test',
                     alignment=alignment,
                     transformations=transformations,
                     balance=False,
                     load_mode='keyframe',
                     max_samples=None)
    # Separate one dataset per video
    datasets_test = utils_dataset.split_data_set_into_videos_lmdb(ds)
    loader_params = {'shuffle': False, 'num_workers': 0, 'worker_init_fn': seed}

    total_pos = len([b for b in ds.keys_ds if b['class_keyframe'] is True])
    total_neg = len([b for b in ds.keys_ds if b['class_keyframe'] is False])
    print_info(f'Testing dataset (fold {fold}) loaded with {len(ds)} samples:', log_path)
    print_info(f'Positive samples: {total_pos}', log_path)
    print_info(f'Negative samples: {total_neg}', log_path)
    print_info(f'Target objects: {", ".join(ds.get_objects())}', log_path, end_block=True)

    # Load module
    model = torch.load(model_path, map_location=device)

    # Freezes everything
    if hasattr(model, 'dissimilarity_module'):
        model.dissimilarity_module.freeze()
    if hasattr(model, 'opening'):
        model.opening.freeze()
    if hasattr(model, 'closing'):
        model.closing.freeze()
    if hasattr(model, 'classification_function'):
        model.classification_function.freeze()
    # Add hooks to obtain the outputs of the net
    hooks_dict = utils_functions.register_hooks(model)
    # Apply testing in each video
    for id_vid, ds in enumerate(datasets_test):
        # Making sure there is only a video at a time
        assert len(set([k['video_name'] for k in ds.keys_ds])) == 1
        vid_basename = ds.keys_ds[0]['video_name']

        if not quiet:
            pos = len([f for f in ds.keys_ds if f['class_keyframe'] is True])
            neg = len([f for f in ds.keys_ds if f['class_keyframe'] is False])
            print_info(
                f'\nEvaluating video {vid_basename} ({len(ds)} frames -> positives: {pos}, negatives: {neg})',
                log_path)

        batch_size = model.temporal_consistency.voting_window if hasattr(
            model, 'temporal_consistency') else 1
        data_loader_validate = DataLoader(ds, **loader_params, batch_size=batch_size)

        count_frames = 0

        metrics_vid = {
            'pred_labels': [],
            'pred_blobs': [],
            'gt_labels': [],
            'gt_bbs': [],
            'computed_metrics': {
                'frame_level': {},
                'pixel_level': {}
            },
            'mean_loss': None
        }
        losses_vid = []
        buffer_frames = {}
        count_samples = 0
        init_frame, central_frame, end_frame = 0, 0, 0

        voting_window = model.temporal_consistency.voting_window if hasattr(
            model, 'temporal_consistency') else None

        if save_videos:
            path_save_videos = os.path.join(dir_save, f'{vid_basename}.avi')
            if not quiet:
                print_info(f'Video output path: {path_save_videos}', log_path)
            writer = imageio.get_writer(path_save_videos, fps=fps, quality=quality, codec='libx264')
        if save_frames:
            dir_save_frames = os.path.join(dir_save, f'{vid_basename}/')
            if not quiet:
                print_info(f'Frames output path: {dir_save_frames}', log_path)
            # Creating folders to separate frames
            os.makedirs(os.path.join(dir_save_frames, 'ref'), exist_ok=True)
            os.makedirs(os.path.join(dir_save_frames, 'tar'), exist_ok=True)
            os.makedirs(os.path.join(dir_save_frames, 'closing'), exist_ok=True)
            os.makedirs(os.path.join(dir_save_frames, 'opening'), exist_ok=True)
            os.makedirs(os.path.join(dir_save_frames, 'dm'), exist_ok=True)
            os.makedirs(os.path.join(dir_save_frames, 'tcm'), exist_ok=True)

        # Evaluate frames
        for batch, (ref_frames, tar_frames, labels_classes, bbs) in enumerate(data_loader_validate):
            # Extract features from the frames with Resnet
            feat_ref = resnet(ref_frames.to(device))
            feat_tar = resnet(tar_frames.to(device))
            # if there is only 1 sample in the batch len(feat_ref.shape) == 3
            if len(feat_ref.shape) == 3:
                feat_ref = feat_ref.unsqueeze(0)
                feat_tar = feat_tar.unsqueeze(0)
                labels_classes = labels_classes.unsqueeze(0)
            samples_batch = len(feat_ref)

            if hasattr(model, 'temporal_consistency'):
                for i in range(samples_batch):
                    buffer_frames[count_samples] = {}
                    buffer_frames[count_samples]['feat_ref'] = feat_ref[i]
                    buffer_frames[count_samples]['feat_tar'] = feat_tar[i]
                    buffer_frames[count_samples]['class'] = labels_classes[i]
                    buffer_frames[count_samples]['frame_ref'] = ref_frames[i]
                    buffer_frames[count_samples]['frame_tar'] = tar_frames[i]
                    buffer_frames[count_samples]['bb'] = bbs[i]
                    count_samples += 1

                init_frame = max(central_frame - voting_window // 2, 0)
                end_frame = min(central_frame + voting_window // 2, len(ds))
                # clean the buffer => remove frames out of the voting window
                ids_to_remove = [i for i in buffer_frames if i < init_frame]
                for i in ids_to_remove:
                    del buffer_frames[i]

                while init_frame in buffer_frames and end_frame in buffer_frames and central_frame < len(
                        ds):
                    # Sets the dictionary with the data to be passed to the network (between init_frame and end_frame)
                    data = {
                        'feat_ref': [],
                        'feat_tar': [],
                        'class': [],
                        'bb': [],
                        'frame_ids': [],
                        'central_frame': central_frame,
                        'frame_ref': [],
                        'frame_tar': []
                    }

                    for i in range(init_frame, end_frame + 1, 1):
                        {data[k].append(v) for k, v in buffer_frames[i].items()}
                        data['frame_ids'].append(i)

                    position_central_frame = data['frame_ids'].index(central_frame)

                    data['feat_ref'] = torch.stack(data['feat_ref'])
                    data['feat_tar'] = torch.stack(data['feat_tar'])

                    outputs = model.inference_validation_test(data)
                    count_frames += 1
                    label_gt = ((data['class'][position_central_frame] * 1.)).to(device)
                    loss = nn.MSELoss()(outputs.squeeze(), label_gt.squeeze())
                    # Compute metrics
                    losses_vid.append(loss.item())
                    output_frame = (hooks_dict['hook_sum_pixels_on'].input[0].squeeze()).to(
                        torch.uint8).cpu().numpy()
                    class_out = (outputs > .5).item()
                    metrics_vid['pred_labels'].append(class_out * 1)
                    metrics_vid['pred_blobs'].append(output_frame)
                    metrics_vid['gt_labels'].append((label_gt.item() == 1) * 1)
                    metrics_vid['gt_bbs'].append(data['bb'][position_central_frame].numpy())
                    # generate frames to be included in the video
                    if save_videos or save_frames:
                        modules = []
                        if hasattr(model, 'dissimilarity_module'): modules.append('DM')
                        if hasattr(model, 'temporal_consistency'): modules.append('TCM')
                        if hasattr(model, 'closing'): modules.append('closing')
                        if hasattr(model, 'opening'): modules.append('opening')
                        if ablation == 'modification1':
                            frames_to_save = get_frames_to_save(hooks_dict, modules, ablation,
                                                                position_central_frame)
                            frames_to_save['frame_id'] = central_frame
                            frames_to_save['ref_frame'] = data['frame_ref'][position_central_frame]
                            frames_to_save['tar_frame'] = data['frame_tar'][position_central_frame]
                            frames_to_save['outputs_model'] = outputs
                            frames_to_save['gt_label'] = data['class'][position_central_frame].item(
                            )
                        elif ablation == 'modification2':
                            frames_to_save = get_frames_to_save(hooks_dict, modules, ablation)
                            frames_to_save['frame_id'] = frame_id
                            frames_to_save['ref_frame'] = ref_frames[sample_id]
                            frames_to_save['tar_frame'] = tar_frames[sample_id]
                            frames_to_save['outputs_model'] = outputs[sample_id]
                            frames_to_save['gt_label'] = data['class'].item()
                        elif ablation == 'modification3':
                            frames_to_save = get_frames_to_save(hooks_dict, modules, ablation,
                                                                position_central_frame)
                            frames_to_save['frame_id'] = central_frame
                            frames_to_save['ref_frame'] = data['frame_ref'][position_central_frame]
                            frames_to_save['tar_frame'] = data['frame_tar'][position_central_frame]
                            frames_to_save['outputs_model'] = outputs
                            frames_to_save['gt_label'] = data['class'][position_central_frame].item(
                            )
                        frames_to_save['CM'] = [class_out]
                        frames_to_save['class_output'] = class_out
                        frames_to_save['rad_open'] = model.opening.se_sigmoid.radius.item(
                        ) if hasattr(model, 'opening') else None
                        frames_to_save['rad_close'] = model.closing.se_sigmoid.radius.item(
                        ) if hasattr(model, 'closing') else None
                    if save_videos:
                        img_strip = create_image_strips(frames_to_save, resnet.STD_IMAGENET,
                                                        resnet.MEAN_IMAGENET)
                        writer.append_data(img_strip)
                    # Save each frame individually as image
                    if save_frames:
                        # Reference frame
                        ref_img = frames_to_save['ref_frame'].cpu()
                        ref_img = (tb_utils.unnormalize(ref_img.unsqueeze(0),
                                                        resnet.STD_IMAGENET,
                                                        resnet.MEAN_IMAGENET,
                                                        one_channel=False) * 255).to(
                                                            torch.uint8).squeeze()
                        Image.fromarray(np.moveaxis(ref_img.numpy(), 0, -1)).save(
                            os.path.join(dir_save_frames, 'ref', f'{init_frame}_ref.png'))
                        # Target frame
                        tar_img = frames_to_save['tar_frame'].cpu()
                        tar_img = (tb_utils.unnormalize(tar_img.unsqueeze(0),
                                                        resnet.STD_IMAGENET,
                                                        resnet.MEAN_IMAGENET,
                                                        one_channel=False) * 255).to(
                                                            torch.uint8).squeeze()
                        Image.fromarray(np.moveaxis(tar_img.numpy(), 0, -1)).save(
                            os.path.join(dir_save_frames, 'tar', f'{init_frame}_tar.png'))
                        Image.fromarray(frames_to_save['DM'].numpy()).save(
                            os.path.join(dir_save_frames, 'dm', f'{init_frame}_dm.png'))
                        Image.fromarray(frames_to_save['TCM'].numpy()).save(
                            os.path.join(dir_save_frames, 'tcm', f'{init_frame}_tcm.png'))
                        if 'opening_output' in frames_to_save:
                            Image.fromarray(frames_to_save['opening_output'].numpy()).save(
                                os.path.join(dir_save_frames, 'opening',
                                             f'{init_frame}_opening.png'))
                        if 'closing_output' in frames_to_save:
                            Image.fromarray(frames_to_save['closing_output'].numpy()).save(
                                os.path.join(dir_save_frames, 'closing',
                                             f'{init_frame}_closing.png'))
                    # Update frames
                    central_frame += 1
                    init_frame = max(central_frame - voting_window // 2, 0)
                    end_frame = min(central_frame + voting_window // 2, len(ds))
                    if end_frame >= len(ds):
                        end_frame = len(ds) - 1
            # No temporal consistency
            else:
                for sample_id in range(samples_batch):
                    # Sets the dictionary with the data to be passed to the network (between init_frame and end_frame)
                    data = {
                        'feat_ref': feat_ref[sample_id].unsqueeze(0),
                        'feat_tar': feat_tar[sample_id].unsqueeze(0),
                        'class': labels_classes[sample_id],
                    }
                    # Pass data through the network
                    outputs = model.inference_validation_test(data)
                    label_gt = ((data['class'] * 1.)).to(device)
                    loss = nn.MSELoss()(outputs.squeeze(), label_gt.squeeze())

                    # Compute metrics
                    losses_vid.append(loss.item())
                    output_frame = (hooks_dict['hook_sum_pixels_on'].input[0].squeeze()).to(
                        torch.uint8).cpu().numpy()
                    class_out = (outputs > .5).item()
                    metrics_vid['pred_labels'].append(class_out * 1)
                    metrics_vid['pred_blobs'].append(output_frame)
                    metrics_vid['gt_labels'].append((label_gt.item() == 1) * 1)
                    metrics_vid['gt_bbs'].append(bbs.numpy().squeeze())
                    frame_id = batch + sample_id

                    # generate frames to be included in the video
                    if save_videos or save_frames:
                        modules = []
                        if hasattr(model, 'dissimilarity_module'): modules.append('DM')
                        if hasattr(model, 'temporal_consistency'): modules.append('TCM')
                        if hasattr(model, 'closing'): modules.append('closing')
                        if hasattr(model, 'opening'): modules.append('opening')
                        frames_to_save = get_frames_to_save(hooks_dict, modules, ablation)
                        frames_to_save['frame_id'] = frame_id
                        frames_to_save['CM'] = [class_out]
                        frames_to_save['gt_label'] = data['class'].item()
                        frames_to_save['ref_frame'] = ref_frames[sample_id]
                        frames_to_save['tar_frame'] = tar_frames[sample_id]
                        frames_to_save['outputs_model'] = outputs[sample_id]
                        frames_to_save['class_output'] = class_out
                        frames_to_save['rad_open'] = model.opening.se_sigmoid.radius.item(
                        ) if hasattr(model, 'opening') else None
                        frames_to_save['rad_close'] = model.closing.se_sigmoid.radius.item(
                        ) if hasattr(model, 'closing') else None
                    if save_videos:
                        img_strip = create_image_strips(frames_to_save, resnet.STD_IMAGENET,
                                                        resnet.MEAN_IMAGENET)
                        writer.append_data(img_strip)
                    # Save each frame individually as image
                    if save_frames:
                        # Reference frame
                        ref_img = frames_to_save['ref_frame'].cpu()
                        ref_img = (tb_utils.unnormalize(ref_img.unsqueeze(0),
                                                        resnet.STD_IMAGENET,
                                                        resnet.MEAN_IMAGENET,
                                                        one_channel=False) * 255).to(
                                                            torch.uint8).squeeze()
                        Image.fromarray(np.moveaxis(ref_img.numpy(), 0, -1)).save(
                            os.path.join(dir_save_frames, 'ref', f'{frame_id}_ref.png'))
                        # Target frame
                        tar_img = frames_to_save['tar_frame'].cpu()
                        tar_img = (tb_utils.unnormalize(tar_img.unsqueeze(0),
                                                        resnet.STD_IMAGENET,
                                                        resnet.MEAN_IMAGENET,
                                                        one_channel=False) * 255).to(
                                                            torch.uint8).squeeze()
                        Image.fromarray(np.moveaxis(tar_img.numpy(), 0, -1)).save(
                            os.path.join(dir_save_frames, 'tar', f'{frame_id}_tar.png'))
                        if 'DM' in frames_to_save:
                            Image.fromarray(frames_to_save['DM'].numpy()).save(
                                os.path.join(dir_save_frames, 'dm', f'{frame_id}_dm.png'))
                        if 'TCM' in frames_to_save:
                            Image.fromarray(frames_to_save['TCM'].numpy()).save(
                                os.path.join(dir_save_frames, 'tcm', f'{frame_id}_tcm.png'))
                        if 'opening_output' in frames_to_save:
                            Image.fromarray(frames_to_save['opening_output'].numpy()).save(
                                os.path.join(dir_save_frames, 'opening', f'{frame_id}_opening.png'))
                        if 'closing_output' in frames_to_save:
                            Image.fromarray(frames_to_save['closing_output'].numpy()).save(
                                os.path.join(dir_save_frames, 'closing', f'{frame_id}_closing.png'))

        # Finished testing / validating one video
        if save_videos:
            writer.close()
        # make sure the amount of positive labels are equivalent to non-empty bounding boxes
        assert sum([1 for b in metrics_vid['gt_bbs']
                    if tuple(b) != (0, 0, 0, 0)]) == sum(metrics_vid['gt_labels'])
        ####################################################################
        # Compute metrics                                                  #
        ####################################################################
        # mean_loss: MSE between output of the classification sigmoid (value between 0 and 1) and the groundtruth label
        metrics_vid['mean_loss'] = np.mean(losses_vid)

        ################################
        # Frame level
        ################################
        # Compute frame-level metric (classification of the frame by the CM)
        # consider predicting labels as 1, if the output of the CM > 0.5 vs. gt labels
        rates = utils_metrics.calculate_TPrate_FPrate(metrics_vid['pred_labels'],
                                                      metrics_vid['gt_labels'])
        tpr, fpr = rates['TP_rate'], rates['FP_rate']
        aux = utils_metrics.get_positives_negatives(metrics_vid['pred_labels'],
                                                    metrics_vid['gt_labels'])
        metrics_vid['computed_metrics']['frame_level'] = {
            'DIS':
            utils_metrics.calculate_DIS(metrics_vid['pred_labels'], metrics_vid['gt_labels']),
            'TPR':
            tpr,
            'FPR':
            fpr,
            'groundtruth_pos':
            aux['groundtruth positives'],
            'groundtruth_neg':
            aux['groundtruth negatives'],
            'sum_tp':
            aux['sum tp'],
            'sum_fp':
            aux['sum fp'],
            'sum_tn':
            aux['sum tn'],
            'sum_fn':
            aux['sum fn'],
            'accuracy':
            utils_metrics.calculate_accuracy(metrics_vid['pred_labels'], metrics_vid['gt_labels'])
        }
        assert metrics_vid['computed_metrics']['frame_level']['accuracy'] == (
            aux['sum tp'] + aux['sum tn']) / (aux['sum tp'] + aux['sum tn'] + aux['sum fp'] +
                                              aux['sum fn'])
        ##################################
        # Compute pixel-level metrics    #
        ##################################
        # First, let's get an image containing the gt bounding box represented by a white area
        gts = {
            'labels': torch.tensor(metrics_vid['gt_labels']),
            'bounding_boxes': metrics_vid['gt_bbs'],
            # 'shape': tar_frames.squeeze().shape
        }
        metrics = utils_metrics.compute_DIS_pixel_level(gts,
                                                        metrics_vid['pred_blobs'],
                                                        alignment=alignment)
        assert metrics['groundtruth_pos'] + metrics['groundtruth_neg'] == 201

        metrics_vid['computed_metrics']['pixel_level'] = {
            'TP': metrics['list_tp'],
            'FP': metrics['list_fp'],
            'FN': metrics['list_fn'],
            'TN': metrics['list_tn'],
            'TPR': metrics['TPR'],
            'FPR': metrics['FPR'],
            'DIS': metrics['DIS'],
            'sum_tp': metrics['sum_tp'],
            'sum_fp': metrics['sum_fp'],
            'sum_tn': metrics['sum_tn'],
            'sum_fn': metrics['sum_fn'],
            'groundtruth_pos': metrics['groundtruth_pos'],
            'groundtruth_neg': metrics['groundtruth_neg'],
            'accuracy': metrics['accuracy'],
        }
        # Print metrics of the video
        if not quiet:
            print_info(f'Computed metrics:', log_path)
            print_info(f'mean_loss: {metrics_vid["mean_loss"]:.4f}', log_path)
            print_info(f'* Frame-level:', log_path)
            print_info(f'\t* TP rate: {metrics_vid["computed_metrics"]["frame_level"]["TPR"]:.4f}',
                       log_path)
            print_info(f'\t* FP rate: {metrics_vid["computed_metrics"]["frame_level"]["FPR"]:.4f}',
                       log_path)
            print_info(f'\t* DIS: {metrics_vid["computed_metrics"]["frame_level"]["DIS"]:.4f}',
                       log_path)
            print_info(
                f'\t* Accuracy: {metrics_vid["computed_metrics"]["frame_level"]["accuracy"]:.4f}',
                log_path)
            print_info(f'* Pixel-level:', log_path)
            print_info(f'\t* TP rate: {metrics_vid["computed_metrics"]["pixel_level"]["TPR"]:.4f}',
                       log_path)
            print_info(f'\t* FP rate: {metrics_vid["computed_metrics"]["pixel_level"]["FPR"]:.4f}',
                       log_path)
            print_info(f'\t* DIS: {metrics_vid["computed_metrics"]["pixel_level"]["DIS"]:.4f}',
                       log_path)
            print_info(
                f'\t* Accuracy: {metrics_vid["computed_metrics"]["pixel_level"]["accuracy"]:.4f}',
                log_path,
                end_block=True,
                sep='-')
        # Gather metrics of the video
        metrics_all_videos[vid_basename] = metrics_vid

    # Append all results in the all_testing_results.pickle
    pickle_results_fp = os.path.join(dir_save, 'all_testing_results.pkl')
    if os.path.isfile(pickle_results_fp):
        existing_results = pickle.load(open(pickle_results_fp, 'rb'))
        metrics_all_videos.update(existing_results)
    pickle.dump(metrics_all_videos, open(pickle_results_fp, 'wb'))


@click.command()
@click.option('--fold', default=1, help='Fold number.', type=click.IntRange(1, 9, clamp=False))
@click.option('--device', default=None, help='GPU device.', type=click.INT)
@click.option('--seed',
              default=123,
              help='Random seed to achieve achieve reproducible results.',
              type=click.INT)
@click.option('--fps', default=5, help='FPS to generate the videos.', type=click.INT)
@click.option('--quality', default=6, help='Quality of the generated videos.', type=click.INT)
@click.option('--ablation',
              default='modification2',
              help='Ablation study.',
              type=click.Choice(
                  ['modification1', 'modification2', 'modification3', 'modification4'],
                  case_sensitive=False))
@click.option('--alignment',
              default='temporal',
              help='Alignment used in the frames.',
              type=click.Choice(['temporal', 'geometric'], case_sensitive=False))
@click.option(
    "--dir_out",
    required=True,
)
@click.option(
    "--dir_pth",
    type=click.Path(exists=False),
    required=True,
)
@click.option('--fp_pkl', type=click.File(), required=True)
@click.option('--save_videos', is_flag=True)
@click.option('--save_frames', is_flag=True)
@click.option('--warnings_on/--warnings_off', default=True)
@click.option('--quiet', is_flag=True)
@click.option('--summarize_on/--summarize_off', default=True)
def main(fold, dir_pth, fp_pkl, ablation, fps, quality, dir_out, alignment, device, seed, quiet,
         save_videos, save_frames, warnings_on, summarize_on):
    os.makedirs(dir_out, exist_ok=True)
    log_path = os.path.join(dir_out, f'testing_results_fold_{fold}.txt')
    init_time = datetime.datetime.now()
    print_info(f'Test initialized at: {init_time.strftime("%Y-%B-%d %H:%M:%S")}\n', log_path)
    print_info(f'Parameters:', log_path, init_block=True)
    print_info(f'fold: {fold}', log_path)
    print_info(f'alignment: {alignment}', log_path)
    print_info(f'ablation: {ablation}', log_path)
    print_info(f'dir_pth: {dir_pth}', log_path)
    print_info(f'fp_pkl: {fp_pkl.name}', log_path)
    print_info(f'fps: {fps}', log_path)
    print_info(f'quality: {quality}', log_path)
    print_info(f'dir_out: {dir_out}', log_path)
    print_info(f'device: {device}', log_path)
    print_info(f'seed: {seed}', log_path)
    print_info(f'quiet: {quiet}', log_path)
    print_info(f'save_videos: {save_videos}', log_path)
    print_info(f'save_frames: {save_frames}', log_path)
    print_info(f'summarize_on: {summarize_on}', log_path)
    print_info(f'warnings_on: {warnings_on}', log_path, end_block=True)

    # Set device
    print_info(f'Attempt to run on device: {device}', log_path)
    if device is not None and torch.cuda.is_available():
        try:
            device = torch.device(f'cuda:{device}')
            torch.cuda.set_device(device)
        except:
            print_info(f'{device} not found', log_path)
            device = torch.device('cpu')

    else:
        print_info(f'{device} not found', log_path)
        device = torch.device('cpu')

    print_info(f'Running on {device}', log_path, end_block=True)

    fp_pkl = str(fp_pkl.name)
    # Load the results.pickle file in the directory
    if not os.path.isfile(str(fp_pkl)):
        print_info(f'\nError: File {fp_pkl} not found.', log_path)
        return
    if not os.path.isdir(dir_pth):
        print_info(f'\nDirectory {dir_pth} was not found.', log_path)
        return

    pkl_file = pickle.load(open(fp_pkl, 'rb'))
    total_val_epochs = len(pkl_file['validation_metrics'])
    print_info(f'A total of {total_val_epochs} validation epochs were found.', log_path)
    # DIS and loss on validation
    DIS_validations = {
        epoch: val_res['summary_validation']['DIS_validation']
        for epoch, val_res in pkl_file['validation_metrics'].items()
    }
    loss_validations = {
        epoch: val_res['summary_validation']['loss_validation']
        for epoch, val_res in pkl_file['validation_metrics'].items()
    }
    # Loss on training
    loss_training = {
        epoch: training_loss['training CM']
        for epoch, training_loss in pkl_file['training_loss'].items()
    }
    # Based on the validation DIS, get the best epoch
    best_val_epoch = min(DIS_validations, key=DIS_validations.get)
    min_val_DIS = DIS_validations[best_val_epoch]
    # Print out
    print_info(f'Best epoch based on the validation DIS: {best_val_epoch}', log_path)
    print_info(f'Epoch {best_val_epoch} reached a validation DIS={min_val_DIS:.4f}', log_path)

    # Find the .pth representing the trained model on the best epoch
    pth_file_name = f'model_epoch_{best_val_epoch}.pth'
    pth_path = utils_functions.find_file(directory=dir_pth, file_name=pth_file_name)
    if not pth_path:
        print_info(
            f'\nError: .pth file ({pth_file_name}) representing the trained model on epoch {best_val_epoch} was not found.',
            log_path)
        return
    print_info(f'Running model {pth_file_name} on the testing set.', log_path, end_block=True)

    # Evaluate the model
    evaluate_model(pth_path,
                   fold,
                   ablation=ablation,
                   alignment=alignment,
                   seed=seed,
                   quiet=quiet,
                   log_path=log_path,
                   dir_save=dir_out,
                   save_videos=save_videos,
                   save_frames=save_frames,
                   fps=fps,
                   quality=quality,
                   device=device)

    # Print all metrics in a single result
    if not summarize_on:
        return

    pickle_results_fp = os.path.join(dir_out, 'all_testing_results.pkl')
    results = pickle.load(open(pickle_results_fp, 'rb'))

    # sort results by video name
    results = {k: results[k] for k in sorted(results.keys())}

    # Compute metrics
    def compute_metrics(type_metric='frame_level'):
        assert type_metric in ['frame_level', 'pixel_level']

        print_info('#' * 60, log_path)
        print_info(f'EVALUATING {type_metric.upper()} METRIC WITH TEMPORAL ALIGNMENT', log_path)
        print_info('#' * 60, log_path)
        print_info('vid sum_tp sum_fp sum_tn sum_fn sum_gt_pos sum_gt_neg TPR FPR DIS', log_path)

        list_tpr, list_fpr, list_dis = [], [], []

        if type_metric == 'frame_level':
            # Variables to compute overall DIS
            sum_tp, sum_fp, sum_tn, sum_fn, sum_groundtruth_pos, sum_groundtruth_neg = 0, 0, 0, 0, 0, 0
            for vid, res in results.items():
                sum_tp += res['computed_metrics'][type_metric]['sum_tp']
                sum_fp += res['computed_metrics'][type_metric]['sum_fp']
                sum_tn += res['computed_metrics'][type_metric]['sum_tn']
                sum_fn += res['computed_metrics'][type_metric]['sum_fn']
                sum_groundtruth_pos += res['computed_metrics'][type_metric]['groundtruth_pos']
                sum_groundtruth_neg += res['computed_metrics'][type_metric]['groundtruth_neg']
                # Compute individual results for the current video
                tp = res['computed_metrics'][type_metric]['sum_tp']
                fp = res['computed_metrics'][type_metric]['sum_fp']
                tn = res['computed_metrics'][type_metric]['sum_tn']
                fn = res['computed_metrics'][type_metric]['sum_fn']
                gt_pos = res['computed_metrics'][type_metric]['groundtruth_pos']
                gt_neg = res['computed_metrics'][type_metric]['groundtruth_neg']
                tpr = tp / (tp + fn) if tp + fn != 0 else 0
                fpr = fp / (fp + tn) if fp + tn != 0 else 0
                dis = np.sqrt((1 - tpr)**2 + fpr**2)
                # Append tpr, fpr and dis to compute the mean
                list_tpr.append(tpr)
                list_fpr.append(fpr)
                list_dis.append(dis)
                print_info(f'{vid} {tp} {fp} {tn} {fn} {gt_pos} {gt_neg} {tpr} {fpr} {dis}',
                           log_path)
            # Compute overall results for frame level
            overall_results = utils_metrics.compute_dis_overall(sum_groundtruth_pos,
                                                                sum_groundtruth_neg, sum_tp, sum_fp,
                                                                sum_tn, sum_fn)
        elif type_metric == 'pixel_level':
            gt_labels, gt_bbs, pred_blobs = [], [], []
            for vid, res_vid in results.items():
                gts_dict = {
                    'labels': torch.tensor(res_vid['gt_labels']),
                    'bounding_boxes': res_vid['gt_bbs']
                }
                res = utils_metrics.compute_DIS_pixel_level(gts_dict,
                                                            res_vid['pred_blobs'],
                                                            alignment='temporal')
                # Compute individual results for the current video
                dis = res['DIS']
                tpr = res['TPR']
                fpr = res['FPR']
                # Append tpr, fpr and dis to compute the mean
                list_tpr.append(tpr)
                list_fpr.append(fpr)
                list_dis.append(dis)
                # Group with previous results so the overall DIS can be computed
                gt_labels += res_vid['gt_labels']
                gt_bbs += res_vid['gt_bbs']
                pred_blobs += res_vid['pred_blobs']
                print_info(
                    f"{vid} {res['sum_tp']} {res['sum_fp']} {res['sum_tn']} {res['sum_fn']} {res['groundtruth_pos']} {res['groundtruth_neg']} {tpr} {fpr} {dis}",
                    log_path)
            # Compute overall results for pixel level
            gts_dict = {
                'labels': torch.tensor(gt_labels),
                'bounding_boxes': gt_bbs,
            }
            overall_results = utils_metrics.compute_DIS_pixel_level(gts_dict,
                                                                    pred_blobs,
                                                                    alignment='temporal')
        # Print results
        print_info('\n', log_path)
        print_info(
            f'Mean values: mean TPR: {sum(list_tpr)/len(list_tpr)} mean FPR: {sum(list_fpr)/len(list_fpr)}  mean DIS: {sum(list_dis)/len(list_dis)} ',
            log_path)
        print_info(
            f"OVERALL \t TPR: {overall_results['TPR']} \t FPR: {overall_results['FPR']} \t DIS: {overall_results['DIS']}",
            log_path)

    # Compute FRAME-LEVEL metrics
    compute_metrics(type_metric='frame_level')
    # Compute OBJECT-LEVEL metrics
    compute_metrics(type_metric='pixel_level')


if __name__ == "__main__":
    main()
    # main(
    #     fold=1,
    #     device=0,
    #     seed=123,
    #     ablation='modification4',
    #     fps=5,
    #     quality=6,
    #     alignment='temporal',
    #     dir_out=
    #     '/home/rafael.padilla/ablation_tcf-lmo/TCF-LMO/testing_logs/testing_results_modification4',
    #     dir_pth=
    #     '/home/rafael.padilla/ablation_tcf-lmo/TCF-LMO/training_logs/training_results_modification4/fold_1_ablation_modification4',
    #     fp_pkl=Path(
    #         '/home/rafael.padilla/ablation_tcf-lmo/TCF-LMO/training_logs/training_results_modification4/fold_1_ablation_modification4/results.pickle'
    #     ),
    #     save_videos=True,
    #     save_frames=True,
    #     warnings_on=True,
    #     quiet=False,
    #     summarize_on=True)

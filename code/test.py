import os
import pickle

import __init_paths__
import click
import imageio
import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import utils.utils_dataset as utils_dataset
import utils.utils_functions as utils_functions
import utils.utils_metrics as utils_metrics
import utils.utils_tensorboard as tb_utils
from lmdb_ds import LMDBDataset
from torch.utils.data import DataLoader
from VDAO_folds.lib.My_Resnet50_reduced import Resnet50_Reduced


def create_image_strips(data_dict, counter, std_val, mean_val, type_model='DM_TCM_MM_CM'):
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
                                                       background='gray',
                                                       scale_factor=1,
                                                       text_area_height=75,
                                                       font_size=15,
                                                       add_border=True)
    ref_tar_strip = np.moveaxis(ref_tar_strip, 0, -1)
    # Create an image with results DM an TCM side by side
    if type_model == 'DM_TCM_MM_CM':
        inputs = torch.cat((data_dict['DM'].unsqueeze(0), data_dict['TCM'].unsqueeze(0),
                            data_dict['opening_output'], data_dict['closing_output']),
                           axis=0)
    elif type_model == 'DM_MM_TCM_CM':
        inputs = torch.cat((data_dict['DM'].unsqueeze(0), data_dict['opening_output'],
                            data_dict['closing_output'], data_dict['TCM'].unsqueeze(0)),
                           axis=0)
    inputs = torch.cat(3 * [inputs.unsqueeze(0)])
    inputs = inputs.permute(1, 0, 2, 3)
    color = 'red' if data_dict["class_output"] else 'dark_green'
    if type_model == 'DM_TCM_MM_CM':
        images_texts = [
            'DM', 'TCM', f'open {data_dict["rad_open"]:.2f}', f'close {data_dict["rad_close"]:.2f}'
        ]
    elif type_model == 'DM_MM_TCM_CM':
        images_texts = [
            'DM', f'open {data_dict["rad_open"]:.2f}', f'close {data_dict["rad_close"]:.2f}', 'TCM'
        ]
    colors_texts = ['black' for i in range(len(images_texts) - 1)]
    colors_texts = colors_texts + [color]
    results_strip = tb_utils.create_image_with_results(inputs,
                                                       images_texts,
                                                       colors_texts,
                                                       background='gray',
                                                       scale_factor=1,
                                                       text_area_height=75,
                                                       font_size=15,
                                                       add_border=True)
    results_strip = np.moveaxis(results_strip, 0, -1)
    # Gathers in a single image all frames target and reference and the results
    H, W, C = ref_tar_strip.shape
    h, w, _ = results_strip.shape
    new_image = np.zeros((h + H, W, C)).astype(np.uint8)
    new_image[0:H, 0:W, :] = ref_tar_strip
    begin = (W - w) // 2
    new_image[H:, begin:begin + w, :] = results_strip
    return new_image


def print_info(text, log_path):
    utils_functions.log(log_path, text, option='a', print_out=True, new_line=True)


@click.command()
@click.option('--fold', default=1, help='Fold number.', type=click.IntRange(1, 9, clamp=False))
@click.option('--device', default=0, help='GPU device.', type=click.INT)
@click.option('--seed',
              default=123,
              help='random seed to achieve achieve reproducible results.',
              type=click.INT)
@click.option('--fps', default=5, help='FPS to generate the videos.', type=click.INT)
@click.option('--quality', default=6, help='Quality of the generated videos.', type=click.INT)
@click.option('--net',
              default='DM_MM_TCM_CM',
              help='Network structure.',
              type=click.Choice(['DM_MM_TCM_CM', 'DM_TCM_MM_CM'], case_sensitive=False))
@click.option(
    "--dir_pth",
    type=click.Path(exists=True),
    required=True,
)
def main(fold, net, device, fps, seed, quality, dir_pth):
    print(f'fold: {fold}')
    print(f'net: {net}')
    print(f'device: {device}')
    print(f'fps: {fps}')
    print(f'quality: {quality}')
    print(f'seed: {seed}')
    print(f'dir_pth: {dir_pth}')

    # Define logging path in the same dir_pth + validate_testing
    dir_log = os.path.join(dir_pth, 'validation_testing')
    os.makedirs(dir_log, exist_ok=True)
    log_path = os.path.join(dir_log, 'logging_test.log')

    # Set device
    torch.cuda.set_device(device)
    device = torch.device(f'cuda:{device}') if torch.cuda.is_available() else torch.device('cpu')

    # Load resnet
    resnet = Resnet50_Reduced(device)
    resnet.freeze()

    # As frames in the LMDB are normalized, lets define the denormalizations
    normalize_transform = transforms.Normalize(mean=resnet.MEAN_IMAGENET, std=resnet.STD_IMAGENET)
    to_tensor_transform = transforms.ToTensor()
    transformations = transforms.Compose([to_tensor_transform, normalize_transform])

    ###################################################
    # Datasets and dataloaders
    ###################################################
    # Validation dataset
    print_info(f'\nDatasets info:\n', log_path)
    dataset_validation = LMDBDataset(fold_number=fold,
                                     type_dataset='validation',
                                     transformations=transformations,
                                     balance=False,
                                     load_mode='keyframe',
                                     max_samples=None)
    datasets_validation = utils_dataset.split_data_set_into_videos_lmdb(dataset_validation)
    total_pos = len([b for b in dataset_validation.keys_ds if b['class_keyframe'] is True])
    total_neg = len([b for b in dataset_validation.keys_ds if b['class_keyframe'] is False])
    print_info(f'Validation dataset (fold {fold}) loaded with {len(dataset_validation)} samples:',
               log_path)
    print_info(f'Positive samples: {total_pos}', log_path)
    print_info(f'Negative samples: {total_neg}', log_path)
    print_info(str(dataset_validation.get_objects()) + '\n', log_path)
    # Testing dataset
    dataset_test = LMDBDataset(fold_number=fold,
                               type_dataset='test',
                               transformations=transformations,
                               balance=False,
                               load_mode='keyframe',
                               max_samples=None)
    datasets_test = utils_dataset.split_data_set_into_videos_lmdb(dataset_test)
    total_pos = len([b for b in dataset_test.keys_ds if b['class_keyframe'] is True])
    total_neg = len([b for b in dataset_test.keys_ds if b['class_keyframe'] is False])
    print_info(f'Testing dataset (fold {fold}) loaded with {len(dataset_test)} samples:', log_path)
    print_info(f'Positive samples: {total_pos}', log_path)
    print_info(f'Negative samples: {total_neg}', log_path)
    print_info(str(dataset_test.get_objects()) + '\n', log_path)
    loader_params = {'shuffle': False, 'num_workers': 0, 'worker_init_fn': seed}

    # define inferene cycle
    inference_cycle = {
        'cycle_name': 'inference',
        'load_mode': 'keyframe',  # in validation or testing, all frames are keyframes
        'loss_func': nn.MSELoss(),
    }

    # Examine *.pth files
    assert os.path.isdir(dir_pth)
    print_info(f'Examining epoch files {dir_pth}', log_path)
    models_paths = utils_functions.get_files_recursively(dir_pth, extension="pth")
    print_info(f'{len(models_paths)} files were found', log_path)

    def validate(cycle,
                 model,
                 dataset,
                 hooks_dict=None,
                 quiet=True,
                 save_video=False,
                 dir_save=None,
                 save_frames=False):
        metrics_all_videos = {}

        # For validation, it is needed to pass one video at a time
        for id_vid_val, ds in enumerate(dataset):
            # Makes sure there is only a video at a time
            assert len(set([k['video_name'] for k in ds.keys_ds])) == 1
            vid_basename = ds.keys_ds[0]['video_name']

            data_loader_validate = DataLoader(ds,
                                              **loader_params,
                                              batch_size=model.temporal_consistency.voting_window)

            if not quiet:
                print_info(f'Evaluating video {vid_basename}', log_path)
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
            voting_window = model.temporal_consistency.voting_window

            if save_video:
                path_save_video = os.path.join(dir_save, f'{vid_basename}.avi')
                writer = imageio.get_writer(path_save_video,
                                            fps=fps,
                                            quality=quality,
                                            codec='libx264')
            if save_frames:
                dir_save_frames = os.path.join(dir_save, f'{vid_basename}')
                os.makedirs(dir_save_frames, exist_ok=True)

            for batch, (ref_frames, tar_frames, labels_classes,
                        bbs) in enumerate(data_loader_validate):
                # features from the frames
                feat_ref = resnet(ref_frames.to(device))
                feat_tar = resnet(tar_frames.to(device))
                # if there is only 1 sample in the batch len(feat_ref.shape) == 3
                if len(feat_ref.shape) == 3:
                    feat_ref = feat_ref.unsqueeze(0)
                    feat_tar = feat_tar.unsqueeze(0)
                    labels_classes = labels_classes.unsqueeze(0)
                samples_batch = len(feat_ref)
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

                    # print(f'central_frame: {central_frame} \t {data["frame_ids"]}')
                    position_central_frame = data['frame_ids'].index(central_frame)

                    data['feat_ref'] = torch.stack(data['feat_ref'])
                    data['feat_tar'] = torch.stack(data['feat_tar'])

                    outputs = model.inference_validation_test(data)
                    count_frames += 1
                    label_gt = ((data['class'][position_central_frame] * 1.)).to(device)
                    loss = cycle['loss_func'](outputs.squeeze(), label_gt.squeeze())
                    losses_vid.append(loss.item())

                    # Compute metrics
                    output_frame = (hooks_dict['hook_sum_pixels_on'].input[0].squeeze()).to(
                        torch.uint8).cpu().numpy()
                    class_out = (outputs > .5).item()
                    metrics_vid['pred_labels'].append(class_out * 1)
                    metrics_vid['pred_blobs'].append(output_frame)
                    metrics_vid['gt_labels'].append((label_gt.item() == 1) * 1)
                    metrics_vid['gt_bbs'].append(data['bb'][position_central_frame].numpy())
                    # generate frames to be included in the video
                    if save_video:
                        # DM -> TCM -> MM -> CM
                        if net == 'DM_TCM_MM_CM':
                            tcm = (hooks_dict['hook_opening'].input[0] * 255).to(torch.uint8).cpu()
                        # DM -> MM -> TCM -> CM
                        elif net == 'DM_MM_TCM_CM':
                            tcm = (hooks_dict['hook_sum_pixels_on'].input[0].squeeze() * 255).to(
                                torch.uint8).cpu()
                        opening_output = (hooks_dict['hook_opening'].output[0] * 255).to(
                            torch.uint8).cpu()
                        closing_output = (hooks_dict['hook_closing'].output[0] * 255).to(
                            torch.uint8).cpu()
                        frames_to_save = {
                            'frame_id':
                            central_frame,
                            'DM': (hooks_dict['hook_dissimilarity'].output[position_central_frame] *
                                   255).to(torch.uint8).cpu(),
                            'TCM':
                            tcm,
                            'CM': [class_out],
                            'gt_label':
                            data['class'][position_central_frame].item(),
                            'ref_frame':
                            data['frame_ref'][position_central_frame],
                            'tar_frame':
                            data['frame_tar'][position_central_frame],
                            'closing_output':
                            closing_output,
                            'opening_output':
                            opening_output,
                            'outputs_model':
                            outputs.unsqueeze(0),
                            'class_output':
                            class_out,
                            'rad_open':
                            model.opening.se_sigmoid.radius.item(),
                            'rad_close':
                            model.closing.se_sigmoid.radius.item()
                        }
                        img_strip = create_image_strips(frames_to_save, position_central_frame,
                                                        resnet.STD_IMAGENET, resnet.MEAN_IMAGENET,
                                                        net)
                        writer.append_data(img_strip)
                    # Save each frame individually as image
                    if save_frames:
                        ref_img = ref_frames[position_central_frame].cpu()
                        ref_img = (tb_utils.unnormalize(ref_img.unsqueeze(0),
                                                        resnet.STD_IMAGENET,
                                                        resnet.MEAN_IMAGENET,
                                                        one_channel=False) * 255).to(
                                                            torch.uint8).squeeze()
                        Image.fromarray(np.moveaxis(ref_img.numpy(), 0, -1)).save(
                            os.path.join(dir_save_frames, f'{init_frame}_ref.png'))
                        tar_img = tar_frames[position_central_frame].cpu()
                        tar_img = (tb_utils.unnormalize(tar_img.unsqueeze(0),
                                                        resnet.STD_IMAGENET,
                                                        resnet.MEAN_IMAGENET,
                                                        one_channel=False) * 255).to(
                                                            torch.uint8).squeeze()
                        Image.fromarray(np.moveaxis(tar_img.numpy(), 0, -1)).save(
                            os.path.join(dir_save_frames, f'{init_frame}_tar.png'))
                        dm = (hooks_dict['hook_dissimilarity'].output[position_central_frame] *
                              255).to(torch.uint8).cpu()
                        Image.fromarray(dm.numpy()).save(
                            os.path.join(dir_save_frames, f'{init_frame}_dm.png'))
                        if net == 'DM_TCM_MM_CM':
                            tcm = (hooks_dict['hook_opening'].input[0] * 255).to(torch.uint8).cpu()
                        elif net == 'DM_MM_TCM_CM':
                            tcm = (hooks_dict['hook_sum_pixels_on'].input[0].squeeze() * 255).to(
                                torch.uint8).cpu()
                        Image.fromarray(tcm.numpy()).save(
                            os.path.join(dir_save_frames, f'{init_frame}_tcm.png'))
                        opening_output = (hooks_dict['hook_opening'].output[0] * 255).to(
                            torch.uint8).cpu()
                        Image.fromarray(opening_output.squeeze().numpy()).save(
                            os.path.join(dir_save_frames, f'{init_frame}_opening.png'))
                        closing_output = (hooks_dict['hook_closing'].output[0] * 255).to(
                            torch.uint8).cpu()
                        Image.fromarray(closing_output.squeeze().numpy()).save(
                            os.path.join(dir_save_frames, f'{init_frame}_closing.png'))

                    # Update frames
                    central_frame += 1
                    init_frame = max(central_frame - voting_window // 2, 0)
                    end_frame = min(central_frame + voting_window // 2, len(ds))

                    if end_frame >= len(ds):
                        end_frame = len(ds) - 1

            # Finished testing / validating one video
            if save_video:
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
            # Compute frame-level metric as I understand it should be computed (classification of the frame by the CM)
            # consider predicting labels as 1, if the output of the CM > 0.5 vs. gt labels
            metrics_vid['computed_metrics']['frame_level'] = {
                'DIS':
                utils_metrics.calculate_DIS(metrics_vid['pred_labels'], metrics_vid['gt_labels']),
                'accuracy':
                utils_metrics.calculate_accuracy(metrics_vid['pred_labels'],
                                                 metrics_vid['gt_labels'])
            }
            ##################################
            # Compute pixel-level metrics
            ##################################
            # First, let's get an image containing the gt bounding box represented by a white area
            gts = {
                'labels': torch.tensor(metrics_vid['gt_labels']),
                'bounding_boxes': metrics_vid['gt_bbs'],
                'shape': tar_frames.squeeze().shape
            }
            gt_masks = utils_functions.get_multiplication_mask(gts,
                                                               no_border_output=True,
                                                               to_device=device)
            gt_masks = gt_masks.squeeze().cpu().numpy().astype(np.uint8) * 255
            acc_bb, acc_tn, acc_fp, acc_fn, acc_tp = [], [], [], [], []
            for pred, gt in zip(metrics_vid['pred_blobs'], gt_masks):
                bb, tn, fp, fn, tp = utils_metrics.frame_eval_pixel(pred, gt)
                acc_bb.append(bb)
                acc_tn.append(tn)
                acc_fp.append(fp)
                acc_fn.append(fn)
                acc_tp.append(tp)
            # 2 - Contabilização de cada vídeo
            # Apenas somar os 201 valores de bb, tn, fp, fn e tp
            bb_sum, tn_sum, fp_sum, fn_sum, tp_sum = sum(acc_bb), sum(acc_tn), sum(acc_fp), sum(
                acc_fn), sum(acc_tp)
            # 3 - Cálculo dos rates
            tp_rate = tp_sum / bb_sum
            fp_rate = fp_sum / len(acc_tp)  # fp_rate = fp_sum / 201
            # 4 - Cálculo do DIS
            dis = np.sqrt((1 - tp_rate)**2 + fp_rate**2)
            accuracy = (tp_sum + tn_sum) / (tp_sum + tn_sum + fp_sum + fn_sum)
            metrics_vid['computed_metrics']['pixel_level'] = {
                'IOU': [],
                'TP': acc_tp,
                'FP': acc_fp,
                'FN': acc_fn,
                'TN': acc_tn,
                'TPR': tp_rate,
                'FPR': fp_rate,
                'DIS': dis,
                'accuracy': accuracy
            }
            ##################################
            # Compute pixel-level metrics
            ##################################
            # Print metrics of the video
            if not quiet:
                print_info(f'{vid_basename} ({count_frames} frames)', log_path)
                print_info(f'mean_loss: {metrics_vid["mean_loss"]:.4f}', log_path)
                print_info(
                    f'frame-level DIS: {metrics_vid["computed_metrics"]["frame_level"]["DIS"]:.4f}',
                    log_path)
                print_info(
                    f'frame-level accuracy: {metrics_vid["computed_metrics"]["frame_level"]["accuracy"]:.4f}',
                    log_path)

                print_info(
                    f'pixel-level DIS: {metrics_vid["computed_metrics"]["pixel_level"]["DIS"]:.4f}',
                    log_path)
                print_info(
                    f'pixel-level accuracy: {metrics_vid["computed_metrics"]["pixel_level"]["accuracy"]:.4f}',
                    log_path)
                print_info('\n', log_path)
            # Save metrics of the video
            metrics_all_videos[vid_basename] = metrics_vid

        ###############################################################
        # Compute general metrics considering all videos
        ###############################################################
        # Cria uma lista gigantesca com todos resultados concatenados
        all_videos_pred_labels = sum(
            [metrics['pred_labels'] for vid, metrics in metrics_all_videos.items()], [])
        all_videos_gt_labels = sum(
            [metrics['gt_labels'] for vid, metrics in metrics_all_videos.items()], [])
        # O DIS overall e accuracy overall do frame level é computado como se fosse um único vídeo gigantesco, por isso que usa o all_videos_gt_labels e o all_videos_gt_labels
        frame_level_DIS_overall = utils_metrics.calculate_DIS(all_videos_pred_labels,
                                                              all_videos_gt_labels)
        frame_level_accuracy_overall = utils_metrics.calculate_accuracy(
            all_videos_pred_labels, all_videos_gt_labels)
        # O DIS overall do pixel-level usa as médias dos TPRs e dos FPRs para calcular o DIS
        mean_TPR = np.mean([
            metrics['computed_metrics']['pixel_level']['TPR']
            for vid, metrics in metrics_all_videos.items()
        ])
        mean_FPR = np.mean([
            metrics['computed_metrics']['pixel_level']['FPR']
            for vid, metrics in metrics_all_videos.items()
        ])
        pixel_level_DIS_overall = np.sqrt((1 - mean_TPR)**2 + (mean_FPR)**2)
        summary_metrics = {
            'mean_loss': np.mean([met['mean_loss'] for vid, met in metrics_all_videos.items()]),
            'frame_level': {
                'DIS_overall': frame_level_DIS_overall,
                'accuracy_overall': frame_level_accuracy_overall
            },
            'pixel_level': {
                'DIS_overall': pixel_level_DIS_overall,
            }
        }

        return {
            'videos': metrics_all_videos,
            'summary': summary_metrics,
        }

    def validate_model(
        model_path,
        dataset,
        quiet=True,
        save_video=False,
        dir_save=None,
        save_frames=False,
    ):
        model = torch.load(model_path, map_location=device)
        # Freezes everything
        model.dissimilarity_module.freeze()
        model.opening.freeze()
        model.closing.freeze()
        model.classification_function.freeze()
        # Add hooks to obtain the outputs of the net
        hooks_dict = utils_functions.register_hooks(model)
        # Apply validation
        loss_validation = validate(inference_cycle,
                                   model,
                                   dataset,
                                   hooks_dict,
                                   quiet=quiet,
                                   save_video=save_video,
                                   dir_save=dir_save,
                                   save_frames=save_frames)
        return loss_validation

    # Create folders to save both validation and testing results
    dir_validation_results = os.path.join(dir_log, 'validation_results')
    dir_testing_results = os.path.join(dir_log, 'testing_results')
    os.makedirs(dir_validation_results, exist_ok=True)
    os.makedirs(dir_testing_results, exist_ok=True)

    # Loop through each model/epoch
    losses_epochs = {}
    for model_path in models_paths:
        # the epoch is expressed in the filename
        epoch = int(os.path.basename(model_path).replace('model_epoch_', '').replace('.pth', ''))

        print_info('#' * 120, log_path)
        print_info(f'\nEvaluating videos epoch {epoch}:', log_path)
        # validate the model with the validation dataset
        validation_results = validate_model(model_path,
                                            datasets_validation,
                                            save_frames=False,
                                            save_video=False,
                                            quiet=False,
                                            dir_save=dir_validation_results)
        # validate the model with the testing dataset
        testing_results = validate_model(model_path,
                                         datasets_test,
                                         save_frames=False,
                                         save_video=False,
                                         quiet=False,
                                         dir_save=dir_testing_results)
        # Holds on the dictionary the obtained loss for this model in this epoch
        losses_epochs[epoch] = {'validation': validation_results, 'testing': testing_results}
        print_info(f'Epoch: {epoch}', log_path)
        print_info(f'mean loss: {validation_results["summary"]["mean_loss"]}', log_path)
        print_info(f'Frame-level:', log_path)
        print_info(
            f'Validation: DIS overall: {validation_results["summary"]["frame_level"]["DIS_overall"]}',
            log_path)
        print_info(
            f'Validation: accucracy: {validation_results["summary"]["frame_level"]["accuracy_overall"]}',
            log_path)
        print_info(
            f'Testing: DIS overall: {testing_results["summary"]["frame_level"]["DIS_overall"]}',
            log_path)
        print_info(
            f'Testing: accucracy: {testing_results["summary"]["frame_level"]["accuracy_overall"]}',
            log_path)
        print_info('\n', log_path)

    # Save pickle containing all validation and testing results among all epochs
    path_to_save = os.path.join(dir_log, f'results_validation_testing.pkl')
    pickle.dump(losses_epochs, open(path_to_save, 'wb'))
    print_info(f'Saving all results at {path_to_save}', log_path)
    # Get epoch with the lowest validation loss
    epochs_losses = {
        epoch: result['validation']['summary']['mean_loss']
        for epoch, result in losses_epochs.items()
    }
    best_epoch = min(epochs_losses, key=epochs_losses.get)
    print_info(f'Epoch with the lowest validation loss: {best_epoch}', log_path)
    print_info(
        f'validation loss: {losses_epochs[best_epoch]["validation"]["summary"]["mean_loss"]}',
        log_path)
    for level in ['frame_level', 'pixel_level']:
        print_info(f'* {level} metrics:', log_path)
        print_info(
            f'validation frame-level DIS: {losses_epochs[best_epoch]["validation"]["summary"][level]["DIS_overall"]}',
            log_path)
        if level == 'frame_level':  # Somente o frame_level tem acurácia
            print_info(
                f'validation frame-level accuracy: {losses_epochs[best_epoch]["validation"]["summary"][level]["accuracy_overall"]}',
                log_path)
        print_info(
            f'testing frame-level overall DIS: {losses_epochs[best_epoch]["testing"]["summary"][level]["DIS_overall"]}',
            log_path)
        if level == 'frame_level':
            print_info(
                f'testing frame-level accuracy: {losses_epochs[best_epoch]["testing"]["summary"][level]["accuracy_overall"]}',
                log_path)
        print_info('\n', log_path)


if __name__ == "__main__":
    main()

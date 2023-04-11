import json
import os
from PIL import Image
import imageio
import torchvision.transforms as transforms
import src.utils.utils_functions as utils_functions
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from glob import glob
from src.utils.VDAO_folds.Resnet50_reduced import Resnet50_Reduced
import pickle
from src.aligned_frames_ds import AlignedFrames
from test import create_image_strips
import src.utils.utils_tensorboard as tb_utils

# Definitions
fold_number = "2"
dir_save = "/home/rafael.padilla/thesis/tcf-lmo/results_luiz_apr_11/" # directory where the results will be saved
refs = f"/nfs/proc/luiz.tavares/VDAO_Database/data/test/ref/fold0{fold_number}/" # directory where the reference frames are
tars = f"/nfs/proc/luiz.tavares/VDAO_Database/data/test/tar/fold0{fold_number}/" # directory where the target frames are
pretrained_model_path = f"/home/rafael.padilla/thesis/tcf-lmo/TCF-LMO/pretrained_models/temporal_alignment_fold_{fold_number}/" # Pretrained downloaded models
device = 0 # GPU number 0, 1, 2 
# End definitions

save_videos = False
save_frames = False
seed = 123
fps = 5
quality = 6
net = "DM_MM_TCM_CM"


fp_pkl = Path(pretrained_model_path) / "results.pickle"
pkl_file = pickle.load(open(fp_pkl, 'rb'))
total_val_epochs = len(pkl_file['validation_metrics'])
print(f'A total of {total_val_epochs} validation epochs were found.')

# DIS and loss on validation
DIS_validations = {
    epoch: val_res['summary_validation']['DIS_validation']
    for epoch, val_res in pkl_file['validation_metrics'].items()
}
# Based on the validation DIS, get the best epoch
best_val_epoch = min(DIS_validations, key=DIS_validations.get)
min_val_DIS = DIS_validations[best_val_epoch]
# Print out
print(f'Best epoch based on the validation DIS: {best_val_epoch}')
print(f'Epoch {best_val_epoch} reached a validation DIS={min_val_DIS:.4f}')
# Find the .pth representing the trained model on the best epoch
model_path = Path(pretrained_model_path) / f'model_epoch_{best_val_epoch}.pth'
assert model_path.exists(), f"File {model_path} not found."

try:
    device = torch.device(f'cuda:{device}')
    torch.cuda.set_device(device)
except:
    print(f'{device} not found')
    device = torch.device('cpu')
metrics_all_videos = {}
print(f'Running on {device}')

# Load resnet
resnet = Resnet50_Reduced(device)
resnet.freeze()

# As frames in the LMDB are normalized, lets define the normalization transformation
normalize_transform = transforms.Normalize(mean=resnet.MEAN_IMAGENET, std=resnet.STD_IMAGENET)
to_tensor_transform = transforms.ToTensor()
transformations = transforms.Compose([to_tensor_transform, normalize_transform])

# Load module
model = torch.load(model_path, map_location=device)

# Freezes everything
model.dissimilarity_module.freeze()
model.opening.freeze()
model.closing.freeze()
model.classification_function.freeze()
# Add hooks to obtain the outputs of the net
hooks_dict = utils_functions.register_hooks(model)

def evaluate_sequential_aligned_frames(ref_paths_frames, tar_paths_frames, vid_basename):
    """Evaluate sequential aligned frames.

    Args:
        ref_paths_frames (List[Path]): List of paths containing sequential reference frames.
        tars (List[Path]): List of paths containing sequential target frames aligned with the reference frames.
        vid_basename (str): Name of the video used to generate output images and video.
    """
    ds = AlignedFrames(ref_frame_paths= ref_paths_frames, tar_frame_paths=tar_paths_frames, transformations=transformations)
    loader_params = {'shuffle': False, 'num_workers': 0, 'worker_init_fn': seed}
    data_loader_validate = DataLoader(ds,
                                        **loader_params,
                                        batch_size=model.temporal_consistency.voting_window)
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

    buffer_frames = {}
    count_samples = 0
    init_frame, central_frame, end_frame = 0, 0, 0
    voting_window = model.temporal_consistency.voting_window

    if save_videos:
        Path(dir_save).mkdir(exist_ok=True)
        path_save_videos = Path(dir_save) / f'{vid_basename}.avi'
        print(f"Video output path: {path_save_videos}")
        writer = imageio.get_writer(path_save_videos, fps=fps, quality=quality, codec='libx264')
    if save_frames:
        dir_save_frames = Path(dir_save) / f'{vid_basename}/'
        Path(dir_save_frames).mkdir(exist_ok=True)
        print(f"Frames output path: {dir_save_frames}")
        # Creating folders to separate frames
        (dir_save_frames / 'ref').mkdir(exist_ok=True)
        (dir_save_frames / 'tar').mkdir(exist_ok=True)
        (dir_save_frames / 'closing').mkdir(exist_ok=True)
        (dir_save_frames / 'opening').mkdir(exist_ok=True)
        (dir_save_frames / 'dm').mkdir(exist_ok=True)
        (dir_save_frames / 'tcm').mkdir(exist_ok=True)

    # Evaluate frames
    for batch, (ref_frames, tar_frames, labels_classes, bbs) in enumerate(data_loader_validate):
        # # Extract features from the frames with Resnet
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

            position_central_frame = data['frame_ids'].index(central_frame)

            data['feat_ref'] = torch.stack(data['feat_ref'])
            data['feat_tar'] = torch.stack(data['feat_tar'])

            outputs = model.inference_validation_test(data)
            count_frames += 1

            # Compute metrics
            output_frame = (hooks_dict['hook_sum_pixels_on'].input[0].squeeze()).to(
                torch.uint8).cpu().numpy()
            class_out = (outputs > .5).item()
            metrics_vid['pred_labels'].append(class_out * 1)
            # generate frames to be included in the video
            if save_videos or save_frames:
                tcm = (hooks_dict['hook_sum_pixels_on'].input[0].squeeze() * 255).to(
                        torch.uint8).cpu()
                opening_output = (hooks_dict['hook_opening'].output[0].squeeze() * 255).to(
                    torch.uint8).cpu()
                closing_output = (hooks_dict['hook_closing'].output[0].squeeze() * 255).to(
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
            if save_videos:
                img_strip = create_image_strips(frames_to_save, position_central_frame,
                                                resnet.STD_IMAGENET, resnet.MEAN_IMAGENET, net)
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
                    os.path.join(dir_save_frames, 'ref', f'{central_frame}_ref.png'))
                # Target frame
                tar_img = frames_to_save['tar_frame'].cpu()
                tar_img = (tb_utils.unnormalize(tar_img.unsqueeze(0),
                                                resnet.STD_IMAGENET,
                                                resnet.MEAN_IMAGENET,
                                                one_channel=False) * 255).to(
                                                    torch.uint8).squeeze()
                Image.fromarray(np.moveaxis(tar_img.numpy(), 0, -1)).save(
                    os.path.join(dir_save_frames, 'tar', f'{central_frame}_tar.png'))
                Image.fromarray(frames_to_save['DM'].numpy()).save(
                    os.path.join(dir_save_frames, 'dm', f'{central_frame}_dm.png'))
                Image.fromarray(frames_to_save['TCM'].numpy()).save(
                    os.path.join(dir_save_frames, 'tcm', f'{central_frame}_tcm.png'))
                Image.fromarray(frames_to_save['opening_output'].numpy()).save(
                    os.path.join(dir_save_frames, 'opening', f'{central_frame}_opening.png'))
                Image.fromarray(frames_to_save['closing_output'].numpy()).save(
                    os.path.join(dir_save_frames, 'closing', f'{central_frame}_closing.png'))

            # Update frames
            central_frame += 1
            init_frame = max(central_frame - voting_window // 2, 0)
            end_frame = min(central_frame + voting_window // 2, len(ds))

            if end_frame >= len(ds):
                end_frame = len(ds) - 1

    if save_videos:
        writer.close()

    path_save_results = Path(dir_save) / f"results_{vid_basename}.json"
    with open(str(path_save_results), 'w') as json_file:
        json.dump(metrics_vid, json_file)

    


for start in np.arange(0, 1206, 201):
    video_paths_refs = [Path(f"{refs}{i:04}.png") for i in range(start, start+201)]
    video_paths_tars = [Path(f"{tars}{i:04}.png") for i in range(start, start+201)]
    start_frame = video_paths_refs[0].stem
    final_frame = video_paths_tars[-1].stem
    vid_basename = f"video_fold0{fold_number}-frames_{start_frame}-{final_frame}"
    print(f"Evaluating {vid_basename}")
    evaluate_sequential_aligned_frames(video_paths_refs, video_paths_tars, vid_basename)

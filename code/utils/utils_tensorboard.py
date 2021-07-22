import os
from collections import OrderedDict, namedtuple
from datetime import datetime
from itertools import product

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont


def unnormalize(img, transf_std, transf_mean, one_channel=False):
    if one_channel:
        transf_std = [transf_std]
        transf_mean = [transf_mean]
    # unnormalize
    img = (img * torch.FloatTensor(transf_std).unsqueeze(1).unsqueeze(1).to(
        img.device)) + torch.FloatTensor(transf_mean).unsqueeze(1).unsqueeze(1).to(img.device)
    return img


def create_image_with_results(images,
                              titles,
                              font_colors,
                              background='gray',
                              scale_factor=1,
                              text_area_height=75,
                              font_size=28,
                              add_border=True):
    # Define text properties
    fontColor = {
        "red": (150, 33, 0),
        "dark_green": (36, 84, 24),
        "green": (33, 150, 0),
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'blue': (0, 0, 200),
        'brown': (160, 82, 45),
        'light blue': (47, 82, 143)
    }
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf",
                              int(font_size * scale_factor),
                              encoding="unic")
    # plot the images in the batch, along with predicted and true labels
    total_images = len(images)
    # Define text area
    sample_image_height, sample_image_width = images.shape[-2:]
    # Make it a 3-channel empty image
    if background == 'white':
        constant_values = 255
        text_area = np.ones((3, text_area_height, sample_image_width), dtype=np.uint8) * 255
    elif background == 'brown':
        text_area = np.ones((3, text_area_height, sample_image_width), dtype=np.uint8)
        text_area[0] = text_area[0] * 160
        text_area[1] = text_area[1] * 82
        text_area[2] = text_area[2] * 45
    elif background == 'light blue':
        text_area = np.ones((3, text_area_height, sample_image_width), dtype=np.uint8)
        text_area[0] = text_area[0] * 92
        text_area[1] = text_area[1] * 111
        text_area[2] = text_area[2] * 143
    elif background == 'gray':
        text_area = np.ones((3, text_area_height, sample_image_width), dtype=np.uint8)
        text_area[0] = text_area[0] * 200
        text_area[1] = text_area[1] * 200
        text_area[2] = text_area[2] * 200
    else:
        constant_values = 0
        text_area = np.zeros((3, text_area_height, sample_image_width), dtype=np.uint8)
    final_image = None
    for idx in range(total_images):
        img = images[idx]
        # It is a 1 channel image
        if img.ndim == 2 or img.shape[0] == 1:
            # Make it a 3-channel image replicating channels
            img = np.stack((img.squeeze(), ) * 3, axis=0)
        # Add text area above the image
        img2 = np.hstack((text_area, img))
        img2 = np.moveaxis(img2, 0, -1)
        img_height, img_width, _ = img2.shape
        # Add border around the image
        if add_border:
            if background == 'brown':
                r = np.pad(img2[:, :, 0], ((0, 5), (5, 5)), mode='constant', constant_values=160)
                g = np.pad(img2[:, :, 1], ((0, 5), (5, 5)), mode='constant', constant_values=82)
                b = np.pad(img2[:, :, 2], ((0, 5), (5, 5)), mode='constant', constant_values=45)
                img2 = np.dstack((r, g, b))
            elif background == 'light blue':
                r = np.pad(img2[:, :, 0], ((0, 5), (5, 5)), mode='constant', constant_values=92)
                g = np.pad(img2[:, :, 1], ((0, 5), (5, 5)), mode='constant', constant_values=111)
                b = np.pad(img2[:, :, 2], ((0, 5), (5, 5)), mode='constant', constant_values=143)
                img2 = np.dstack((r, g, b))
            elif background == 'gray':
                r = np.pad(img2[:, :, 0], ((0, 5), (5, 5)), mode='constant', constant_values=200)
                g = np.pad(img2[:, :, 1], ((0, 5), (5, 5)), mode='constant', constant_values=200)
                b = np.pad(img2[:, :, 2], ((0, 5), (5, 5)), mode='constant', constant_values=200)
                img2 = np.dstack((r, g, b))
            else:
                img2 = np.pad(img2, ((0, 0), (5, 5), (0, 0)),
                              mode='constant',
                              constant_values=constant_values)
        # Create PIL object to add text
        im = Image.fromarray(img2)
        (width, height) = (int(im.width * scale_factor), int(im.height * scale_factor))
        im = im.resize((width, height))
        img_width, img_height = im.size
        # Create draw object and text properties
        draw = ImageDraw.Draw(im)
        w1, h1 = draw.textsize(titles[idx], font=font)
        draw.text(((img_width - w1) / 2, ((text_area_height * scale_factor) - h1) / 2),
                  titles[idx],
                  fontColor[font_colors[idx]],
                  font=font)
        del draw
        if final_image is None:
            final_image = im
        else:
            final_image = np.hstack((final_image, np.array(im)))
    return np.moveaxis(np.array(final_image), -1, 0)


def create_training_img_from_binarized(img, desired_output_shape):
    resized_img = torch.zeros(desired_output_shape)
    img = (img.squeeze(1) * 255).cpu().numpy().astype(np.uint8)
    for i, batch in enumerate(img):
        resized_img[i] = torch.tensor(
            np.array(
                Image.fromarray(batch).resize(
                    desired_output_shape[2:][::-1]).convert(mode="RGB"))).permute(2, 0, 1)
    return resized_img


def save_trainable_values(tensor_board, network, step, prefix_var='', histogram=True):
    # Add histograms (an also will appear in the distributions tab)
    trainable_values = network.get_trainable_values()
    for module_name, dict_vars_module in trainable_values.items():
        for var_name, value in dict_vars_module.items():
            save_name = f'{module_name} {var_name}'
            value = value.mean() if isinstance(value, np.ndarray) else value
            if histogram:
                tensor_board.add_histogram(save_name, value, step)
            else:
                tensor_board.add_scalar(save_name, value, step)
    tensor_board.close()


def generate_images_in_grid(binary_img, ref_frames, tar_frames, desired_output_shape, std_imagenet,
                            mean_imagenet):
    binarized_distance_as_img = create_training_img_from_binarized(binary_img, desired_output_shape)
    # Create image grid with tar, ref and binarized and send data to tensorboard
    image_grid = torchvision.utils.make_grid(torch.cat(
        (unnormalize(ref_frames, std_imagenet, mean_imagenet, one_channel=False),
         unnormalize(tar_frames, std_imagenet, mean_imagenet,
                     one_channel=False), binarized_distance_as_img), 0),
                                             nrow=5)
    return image_grid


def get_names_of_classes(classes):
    return ['anomalous' if c else 'not anomalous' for c in classes]


def get_strips_intermediate_images(**kwargs):
    ###########################################################################################
    # Create a figure showing samples (ref, tar and binary), predicted classes and percentage #
    ###########################################################################################
    ref_img = kwargs['ref_img']
    tar_img = kwargs['tar_img']
    ##########################################################################################
    # Store a figure showing morphology results (closing, opening), with total pixels on     #
    ##########################################################################################
    dissim_module_img = kwargs['dissimilarity_output']
    temp_consist_module_img = kwargs['temporal_consistency_output']
    opening_img = kwargs['opening_output']
    closing_img = kwargs['closing_output']
    batch_size = len(opening_img)
    sf_img_results = 2
    # From propabilities, get classes
    colors_based_results = [
        'green' if ft == pre else 'red'
        for ft, pre in zip(kwargs['gt_classes'], kwargs['preds_classes'])
    ]
    ref_strip = create_image_with_results(ref_img,
                                          titles=['reference frame'] * batch_size,
                                          font_colors=colors_based_results,
                                          scale_factor=.5)
    tar_strip = create_image_with_results(
        tar_img,
        titles=[f'target ({c})' for c in get_names_of_classes(kwargs['gt_classes'])],
        font_colors=colors_based_results,
        scale_factor=.5)
    dissim_module_strip = create_image_with_results(dissim_module_img,
                                                    titles=['DM'] * batch_size,
                                                    font_colors=colors_based_results,
                                                    scale_factor=sf_img_results,
                                                    text_area_height=30,
                                                    font_size=10)
    temp_consist_module_strip = create_image_with_results(temp_consist_module_img,
                                                          titles=['TCM'] * batch_size,
                                                          font_colors=colors_based_results,
                                                          scale_factor=sf_img_results,
                                                          text_area_height=30,
                                                          font_size=10)
    rad_open = kwargs['rad_open']
    thresh_open = kwargs['thresh_open']
    open_strip = create_image_with_results(
        opening_img,
        titles=[f'opened({rad_open:.4f})\nthresh({thresh_open:.4f})'] * batch_size,
        font_colors=colors_based_results,
        scale_factor=sf_img_results,
        text_area_height=30,
        font_size=10)
    rad_close = kwargs['rad_close']
    thresh_close = kwargs['thresh_close']
    close_strip = create_image_with_results(
        closing_img,
        titles=[f'closed({rad_close:.4f})\nthresh({thresh_close:.4f})'] * batch_size,
        font_colors=colors_based_results,
        scale_factor=sf_img_results,
        text_area_height=30,
        font_size=10)
    # Strip  with amount of pixels 'on' after closing
    max_val_pixels = closing_img.shape[-1] * closing_img.shape[-2] * 255
    pixels_on = closing_img.sum(axis=(-1, -2))
    pixels_on_text = [f'pixels on: {100*i.item()/max_val_pixels:.2f} %' for i in pixels_on]
    pixels_on_strip = create_image_with_results(
        torch.ones(batch_size, 1, 2, closing_img.shape[-1], dtype=torch.uint8) * 255,
        titles=pixels_on_text,
        font_colors=colors_based_results,
        scale_factor=sf_img_results,
        text_area_height=30,
        font_size=10)
    # Strip  with output of the network. Represents the proportion (%) of pixels 'on'
    outputs_model = [f'output: {i:.2f}%' for i in kwargs['outputs_model']]
    text_outputs_model = create_image_with_results(
        torch.ones(batch_size, 1, 2, closing_img.shape[-1], dtype=torch.uint8) * 255,
        titles=outputs_model,
        font_colors=colors_based_results,
        scale_factor=sf_img_results,
        text_area_height=30,
        font_size=10)
    # Strip with final result
    preds_classes = get_names_of_classes(kwargs['preds_classes'])
    gt_classes = get_names_of_classes(kwargs['gt_classes'])
    final_result = [
        f'{gt_classes[i]}->{preds_classes[i]}' for i, pred_class in enumerate(preds_classes)
    ]
    final_result_strip = create_image_with_results(
        torch.ones(batch_size, 1, 2, closing_img.shape[-1], dtype=torch.uint8) * 255,
        titles=final_result,
        font_colors=colors_based_results,
        scale_factor=sf_img_results,
        text_area_height=30,
        font_size=10)
    # Create unique strip with input frames, morphology and texts
    input_frames = np.hstack((ref_strip, tar_strip))
    res_morphology = np.hstack(
        (dissim_module_strip, temp_consist_module_strip, open_strip, close_strip))
    padding_hor = res_morphology.shape[2] - input_frames.shape[2]
    input_frames = np.pad(input_frames,
                          ((0, 0), (0, 0),
                           (int(padding_hor / 2), padding_hor - int(padding_hor / 2))),
                          mode='constant',
                          constant_values=255)
    # Create a single strip with all images
    return np.hstack(
        (input_frames, res_morphology, pixels_on_strip, text_outputs_model, final_result_strip))

import fnmatch
import os

import cv2
import numpy as np
import PIL.Image as Image
import torch
from PIL import ImageDraw, ImageFont
from scipy import ndimage
from skimage import measure

from .hook_net import Hook

VDAO_original_resolution = {'height': 720, 'width': 1280}
VDAO_half_resolution = {
    'height': VDAO_original_resolution['height'] // 2,
    'width': VDAO_original_resolution['width'] // 2
}
VDAO_half_resolution_no_border = {
    'height': int(.9 * VDAO_half_resolution['height']),
    'width': int(.9 * VDAO_half_resolution['width'])
}
resnet_output_resolution = {'height': 90, 'width': 160}
# When the geometric aligment is used, the resolution reduced to 90% of its original
resnet_output_resolution_no_border = {
    'height': int(resnet_output_resolution['height'] * .90),
    'width': int(resnet_output_resolution['width'] * .90)
}


def add_bb_into_image(image, boundingBox, color, thickness, label=None):
    r = int(color[0])
    g = int(color[1])
    b = int(color[2])

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontThickness = 1

    xIn = boundingBox[0]
    yIn = boundingBox[1]
    cv2.rectangle(image, (boundingBox[0], boundingBox[1]), (boundingBox[2], boundingBox[3]),
                  (b, g, r), thickness)
    # Add label
    if label is not None:
        # Get size of the text box
        (tw, th) = cv2.getTextSize(label, font, fontScale, fontThickness)[0]
        # Top-left coord of the textbox
        (xin_bb, yin_bb) = (xIn + thickness, yIn - th + int(12.5 * fontScale))
        # Checking position of the text top-left (outside or inside the bb)
        if yin_bb - th <= 0:  # if outside the image
            yin_bb = yIn + th  # put it inside the bb
        r_Xin = xIn - int(thickness / 2)
        r_Yin = yin_bb - th - int(thickness / 2)
        # Draw filled rectangle to put the text in it
        cv2.rectangle(image, (r_Xin, r_Yin - thickness),
                      (r_Xin + tw + thickness * 3, r_Yin + th + int(12.5 * fontScale)), (b, g, r),
                      -1)
        cv2.putText(image, label, (xin_bb, yin_bb), font, fontScale, (0, 0, 0), fontThickness,
                    cv2.LINE_AA)
    return image


def separate_into_blobs(binary_image):
    assert binary_image.max() <= 1
    assert binary_image.min() >= 0
    ret = []
    # if image is in graylevel, binarize it
    binary_image = (binary_image > .5) * 1
    blobs_labels = measure.label(binary_image, background=0)
    for label in range(len(ndimage.find_objects(blobs_labels, max_label=50))):
        if label == 0:  # blob label 0 is always a background image
            continue
        blob = (blobs_labels == label) * 1
        if blob.sum() == 0:
            continue
        ret.append(blob)
    return ret


def log(filepath, text, option='a', print_out=True, new_line=False):
    with open(filepath, option) as f:
        f.write(text)
        if new_line:
            f.write('\n')
    if print_out:
        print(text)


def find_file(directory, file_name, match_extension=True):
    if os.path.isdir(directory) is False:
        return None
    for dirpath, dirnames, files in os.walk(directory):
        for f in files:
            f1 = os.path.basename(f)
            f2 = file_name
            if not match_extension:
                f1 = os.path.splitext(f1)[0]
                f2 = os.path.splitext(f2)[0]
            if f1 == f2:
                return os.path.join(dirpath, os.path.basename(f))
    return None


def get_files_recursively(directory, extension="*"):
    if '.' not in extension:
        extension = '*.' + extension
    files = [
        os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(directory)
        for f in fnmatch.filter(files, extension)
    ]
    # Disconsider hidden files, such as .DS_Store in the MAC OS
    ret = [f for f in files if not os.path.basename(f).startswith('.')]
    return ret


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
                              background='white',
                              scale_factor=1,
                              text_area_height=75,
                              font_size=28,
                              add_border=True):
    if images.dtype != np.uint8:
        images = (255 * images).astype(np.uint8)

    # Define text properties
    fontColor = {
        "red": (255, 0, 0),
        'dark_green': (36, 84, 24),
        "green": (0, 255, 0),
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'blue': (0, 0, 200),
        'brown': (160, 82, 45)
    }
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf",
                              int(font_size * scale_factor),
                              encoding="unic")
    # Define text area
    if len(images.shape) == 4:
        total_images, sample_image_height, sample_image_width, channels = images.shape
    elif len(images.shape) == 3:
        total_images, sample_image_height, sample_image_width = images.shape
    # sample_image_height, sample_image_width = images.shape[-2:]
    # Make it a 3-channel empty image
    if background == 'white':
        constant_values = 255
        text_area = np.ones((text_area_height, sample_image_width, 3), dtype=np.uint8) * 255
    elif background == 'brown':
        text_area = np.ones((text_area_height, sample_image_width, 3), dtype=np.uint8)
        text_area[0] = text_area[0] * 160
        text_area[1] = text_area[1] * 82
        text_area[2] = text_area[2] * 45
    else:
        constant_values = 0
        text_area = np.zeros((text_area_height, sample_image_width, 3), dtype=np.uint8)
    final_image = None
    for idx in range(total_images):
        img = images[idx]
        # It is a 1 channel image
        if img.ndim == 2 or img.shape[0] == 1:
            # Make it a 3-channel image replicating channels
            img = np.stack((img.squeeze(), ) * 3, axis=2)
        # Add text area above the image
        img2 = np.vstack((text_area, img))
        # img2 = np.moveaxis(img2, 0, -1)
        img_height, img_width, _ = img2.shape
        # Add border around the image
        if add_border:
            if background == 'brown':
                r = np.pad(img2[:, :, 0], ((0, 5), (5, 5)), mode='constant', constant_values=160)
                g = np.pad(img2[:, :, 1], ((0, 5), (5, 5)), mode='constant', constant_values=82)
                b = np.pad(img2[:, :, 2], ((0, 5), (5, 5)), mode='constant', constant_values=45)
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
    return final_image


def register_hooks(model):
    # Register hooks
    return {
        'hook_dissimilarity': Hook(module=model.dissimilarity_module),
        'hook_opening': Hook(module=model.opening),
        'hook_closing': Hook(module=model.closing),
        'hook_sum_pixels_on': Hook(module=model.sum_pixels_on),
        'hook_new_scale_pixels_on': Hook(module=model.change_scale_for_classification),
    }


def reposition_bb(bounding_box, img_h_w, border_percentage_removal):
    # bounding_box -> (x, y, x2, y2)
    # img_h_w -> height and width of original image, where bb is present and with the border
    if bounding_box is None:
        return None
    if len(img_h_w) == 3:
        H, W, _ = img_h_w
    else:
        H, W = img_h_w
    rx = W * border_percentage_removal
    ry = H * border_percentage_removal
    nh, nw = int(H - (2 * ry)), int(W - (2 * rx))
    x, y, x2, y2 = bounding_box
    # recompute coordinates of the bb disconsidering the removed percentage of the image
    nx1 = max(int(x - rx), 0)
    ny1 = max(int(y - ry), 0)
    nx2 = min(max(int(x2 - rx), 0), nw)
    ny2 = min(max(int(y2 - ry), 0), nh)
    # of new x1 or new y1 are out of the new image, bounding box is not visible anymore
    if nx1 > nw or ny1 > nh or (nx2 - nx1 == 0) or (ny2 - ny1 == 0):
        return None
    return (nx1, ny1, nx2, ny2)


def get_multiplication_mask(gt, alignment='geometric', to_device=None):
    if to_device is None:
        to_device = torch.device('cpu')
    # Depending on the alignemnt of the database, the output of the network is different
    if alignment == 'geometric':
        resolution_height_output, resolution_width_output = resnet_output_resolution_no_border[
            'height'], resnet_output_resolution_no_border['width']
        resolution_height_gt, resolution_width_gt = VDAO_half_resolution_no_border[
            'height'], VDAO_half_resolution_no_border['width']
    elif alignment == 'temporal':
        resolution_height_output, resolution_width_output = resnet_output_resolution[
            'height'], resnet_output_resolution['width']
        resolution_height_gt, resolution_width_gt = VDAO_half_resolution[
            'height'], VDAO_half_resolution['width']
    else:
        raise Exception('parameter alignment can be geometric or temporal only')
    # get amount of frames (batch)
    batch = len(gt['bounding_boxes'])
    dx = resolution_width_output / resolution_width_gt
    dy = resolution_height_output / resolution_height_gt
    # Creat N masks with 1's with the same resolution of the output of the network (n=batches)
    masks_ones = torch.zeros(
        (batch, 1, resolution_height_output, resolution_width_output)).to(to_device)
    # Para cada imagem, coloca 1 na região do bounding box
    for i, label in enumerate(gt['labels']):
        if label.item() == 1:
            bb = gt['bounding_boxes'][i]
            x1 = int(dx * bb[0].item())
            x2 = int(dx * bb[2].item())
            y1 = int(dy * bb[1].item())
            y2 = int(dy * bb[3].item())
            # Replace area of the bounding box of image i with 1's
            bb_height = max(y2 - y1, 1)
            bb_width = max(x2 - x1, 1)
            masks_ones[i][0][y1:y1 + bb_height, x1:x1 + bb_width] = torch.ones(
                (bb_height, bb_width))
            # Image.fromarray(255 *
            #                 masks_ones[i].squeeze().cpu().detach().numpy().astype(np.uint8)).show()
            # as label == 1, make sure the produced mask contains at least one white pixel
            assert masks_ones[i].squeeze().cpu().detach().numpy().astype(np.uint8).max() == 1
            # Image.fromarray(
            #     255 *
            #     masks_ones[i].squeeze().cpu().detach().numpy().astype(np.uint8)).save(f'{i}.png')
        else:
            # Make sure bounding box is (0,0,0,0) and label is False
            assert sum(gt['bounding_boxes'][i]).item() == label.item()
    return masks_ones


def get_metrics_on_bb_mask(x, masks_bb_ones, device, threshold=0.5):
    x = x.to(device)
    x_thresholded = (x > threshold) * 1.
    # Get threshold to segregate what is anomalous pixel or not
    pixels_in_image = x.shape[-1] * x.shape[-2]
    # Check if there isnt anomaly to be detected
    no_anomalies_to_be_detected = masks_bb_ones.sum(axis=(-1, -2, -3)) == 0
    # Verify if detected everything as anomaly
    all_pixels_detected_as_anomalous = x_thresholded.sum(axis=(-1, -2, -3)) == pixels_in_image
    # Verify if all ground-truth is anomalous
    all_pixels_gt_anomalous = masks_bb_ones.sum(axis=(-1, -2, -3)) == pixels_in_image
    # Verify if no pixel was detected as anomalous
    no_anomalous_pixels_detected = x_thresholded.sum(axis=(-1, -2, -3)) == 0

    num_images = len(x)
    for i in range(num_images):
        coord_i_diff, coord_j_diff, coord_i_eq, coord_j_eq = None, None, None, None
        equal_coords = torch.where(
            x_thresholded[i, :, :, :].squeeze() == masks_bb_ones[i, :, :, :].squeeze())
        # there is a position where the values are the same
        if len(equal_coords[0]):
            coord_i_eq, coord_j_eq = equal_coords[0][0].item(), equal_coords[1][0].item()
        diff_coords = torch.where(
            x_thresholded[i, :, :, :].squeeze() != masks_bb_ones[i, :, :, :].squeeze())
        # there is a position where the values are different
        if len(diff_coords[0]) != 0:
            coord_i_diff, coord_j_diff = diff_coords[0][0].item(), diff_coords[1][0].item()

        match_cases = 0
        # gt without anomaly
        if no_anomalies_to_be_detected[i].item():
            # Case 5: No anomaly to be detected, but all pixels were detected as anomalous
            if all_pixels_detected_as_anomalous[i]:
                # Case 5: No anomaly to be detected, but all pixels were detected as anomalous
                # Inverte pixels diferentes
                masks_bb_ones[i, :, coord_i_diff, coord_j_diff] = 1.
                x[i, :, coord_i_diff, coord_j_diff] *= 0.
                match_cases += 1
            else:
                # No anomaly to detect, but it did not detect all pixels as anomalous
                # Case 1: No anomaly to detect and no anomaly to was detected &
                # Cases 2, 3 e 4: No anomaly to detect and some pixels were detected as anomalous
                # Transform equal pixels (non anomalous) as 1 in both images
                masks_bb_ones[i, :, coord_i_eq, coord_j_eq] = 1.
                x[i, :, coord_i_eq, coord_j_eq] += 1. - x[i, :, coord_i_eq, coord_j_eq]
                match_cases += 1
        # gt with anomaly
        else:
            # Case 9: the whole gt is anomalous, but it did not detect any pixel as anomalous
            if all_pixels_gt_anomalous[i] and no_anomalous_pixels_detected[i]:
                # Invert different pixels
                masks_bb_ones[i, :, coord_i_diff, coord_j_diff] = 0.
                x[i, :, coord_i_diff, coord_j_diff] += 1. - x[i, :, coord_i_diff, coord_j_diff]
                match_cases += 1
            # Case 7: the whole gt is anomalous and all pixels were detected as anomalous
            if all_pixels_gt_anomalous[i] and all_pixels_detected_as_anomalous[i]:
                # Invert equal pixels
                masks_bb_ones[i, :, coord_i_eq, coord_j_eq] = 0.
                x[i, :, coord_i_eq, coord_j_eq] *= 0.
                match_cases += 1
            # Case 6: gt is not completely anomalous and all pixels were detected as anomalous
            if not all_pixels_gt_anomalous[i] and all_pixels_detected_as_anomalous[i]:
                # Puts the different pixel (non anomalous) as 1 in the gt, and inverts the different pixel in the detection
                masks_bb_ones[i, :, coord_i_diff, coord_j_diff] = 1.
                x[i, :, coord_i_diff, coord_j_diff] *= 0.
                match_cases += 1
            # Case 8: gt not completely anomalous and no pixel was detected as anomalous
            if not all_pixels_gt_anomalous[i] and no_anomalous_pixels_detected[i]:
                # Puts the equal pixel (non anomalous) as anomalous in both gt and detection
                masks_bb_ones[i, :, coord_i_eq, coord_j_eq] = 1.
                x[i, :, coord_i_eq, coord_j_eq] += 1. - x[i, :, coord_i_eq, coord_j_eq]
                match_cases += 1

        # Make sure that it entered in at most 1 case
        assert match_cases <= 1
        # Make sure that the gt and detections there are no negative pixel and no pixel with value > 1
        assert (x[i, :, :, :] > 1).sum() + (x[i, :, :, :] < 0).sum() == 0
        assert (masks_bb_ones[i, :, :, :] > 1).sum() + (masks_bb_ones[i, :, :, :] < 0).sum() == 0

    pixels_on_inside_bb = masks_bb_ones * x
    pixels_on_outside_bb = (1 - masks_bb_ones) * x
    pixels_off_inside_bb = masks_bb_ones * (1 - x)
    pixels_off_outside_bb = (1 - masks_bb_ones) * (1. - x)
    return {
        'TP': pixels_on_inside_bb.sum(axis=(-1, -2)),
        'FP': pixels_on_outside_bb.sum(axis=(-1, -2)),
        'FN': pixels_off_inside_bb.sum(axis=(-1, -2)),
        'TN': pixels_off_outside_bb.sum(axis=(-1, -2))
    }


def calculate_norm_mcc(output, gt, device, alignment='geometric', threshold=0.5):
    # Threshold defines if a pixel is 'on' or 'off'
    # As DM image is almost binary, let us consider threshold = 0.5
    # Get multiplication mask
    multiplication_mask = get_multiplication_mask(gt, alignment=alignment, to_device=device)
    # Get metrics (TP, FP, TN, FN) based on the masks and results
    results = get_metrics_on_bb_mask(output.clone(),
                                     multiplication_mask,
                                     threshold=threshold,
                                     device=device)
    # Get results
    TP = results['TP']
    FP = results['FP']
    TN = results['TN']
    FN = results['FN']
    # Compute MCC
    mcc = ((TP * TN) - (FP * FN)) / torch.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    assert sum([m.item() < -1 or m.item() > 1 for m in mcc]) == 0
    # MCC originally has values between -1 and 1. So, we need to normalize it between 0 and 1
    norm_mcc = (mcc + 1) / 2
    assert sum([m.item() < 0 or m.item() > 1 for m in norm_mcc]) == 0
    return norm_mcc


def all_items_are_equal(list_items):
    for idx, i in enumerate(list_items):
        if list_items[idx] != list_items[idx - 1]:
            return False
    return True


def calculate_best_window_temporal_consistency(output, gt, device, alignment, threshold=0.5):
    multiplication_mask = get_multiplication_mask(gt, alignment=alignment, to_device=device)
    multiplication_mask = multiplication_mask[gt['middle_id']]
    shape_mult_mask = multiplication_mask.shape
    results2 = get_metrics_on_bb_mask(
        torch.stack(list(output.values())).unsqueeze(1).clone(),  # Include batch, channel
        multiplication_mask.unsqueeze(0).expand(
            len(output), shape_mult_mask[0], shape_mult_mask[1],
            shape_mult_mask[2]).clone(),  # Include batch, channel
        threshold=threshold,
        device=device)
    # Get results
    TP = results2['TP']
    FP = results2['FP']
    TN = results2['TN']
    FN = results2['FN']
    # Compute MCC
    mcc = ((TP * TN) - (FP * FN)) / torch.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    norm_mcc = (mcc + 1) / 2
    ret2 = {window: norm_mcc[idx].unsqueeze(0) for idx, window in enumerate(output.keys())}
    return ret2

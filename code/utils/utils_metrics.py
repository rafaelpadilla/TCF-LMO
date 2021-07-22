import cv2
import numpy as np


def frame_eval_pixel(vid, gt):
    # vid - predicted frame
    # gt  - ground truth
    bb = 0  # has bounding box
    bb, tn, fp, fn, tp = 0, 0, 0, 0, 0
    # if ground truth has an object
    if 255 in gt:
        bb = 1
        # region inside bb
        ins_region = cv2.threshold(gt, 200, 1, cv2.THRESH_BINARY)[1]
        out_region = cv2.threshold(gt, 50, 1, cv2.THRESH_BINARY_INV)[1]
        # Checking blobs inside BB
        if (vid * ins_region).sum() > 0:
            tp = 1
        else:
            fn = 1
        # Counting blobs
        blobs_img = cv2.connectedComponents(vid.astype('uint8'))[1]
        ins_blobs = np.unique(ins_region * blobs_img)
        out_blobs = np.unique(out_region * blobs_img)
        extra_blobs = [k for k in out_blobs if k not in ins_blobs]
        # Has extra blobs
        if len(extra_blobs) > 0:
            fp = 1
        else:
            fp = 0
    # If the frame doesn't have a bounding box:
    else:
        if np.sum(vid) > 0:
            fp = 1
        else:
            tn = 1
    return bb, tn, fp, fn, tp


def calculate_IOU_blob_and_bb(binary_blob_img, gt_mask):
    # Create an empty image with the same resolution as the blob img
    assert binary_blob_img.shape == gt_mask.shape  # make sure they have the same resolution
    assert binary_blob_img.max() <= 1  # makes sure it is binary
    assert gt_mask.max() <= 1  # make sure it is binary
    res_sum = binary_blob_img + gt_mask
    intersection = (res_sum == 2).sum()
    union = (res_sum > 0).sum()
    return {'iou': intersection / union, 'intersection': intersection, 'union': union}


def calculate_accuracy(pred_labels, gt_labels):
    if len(pred_labels) == 0 and len(gt_labels) == 0:
        return 0
    # Calculate accuracy
    total_correct = sum([(pred == gt) for pred, gt in zip(pred_labels, gt_labels)])
    return total_correct / len(gt_labels)


def calculate_DIS(pred_labels, gt_labels):
    # Get amount of positives and negatives
    amount_pos = (np.array(gt_labels) == 1).sum()
    amount_neg = (np.array(gt_labels) == 0).sum()
    # Get TP and FP
    TP, FP = 0, 0
    for pred, gt in zip(pred_labels, gt_labels):
        # If frame was considered as having a lost object
        if pred == 1:
            # Predicted an object and there was an object
            if gt == 1:
                TP += 1
            # Predicted an object but there was no object
            else:
                FP += 1
    # TP_rate: TP / (TP+FN) and TP+FN = the total number of positives
    TP_rate = TP / amount_pos if amount_pos != 0 else 0.
    # FP_rate: FP / (FP+TN) and FP+TN = the total number of negatives
    FP_rate = FP / amount_neg if amount_neg != 0 else 0.
    # Calculate DIS
    return np.sqrt((1 - TP_rate)**2 + FP_rate**2)


def calculate_TPrate_FPrate(pred_labels, gt_labels):
    # Get amount of positives and negatives
    amount_pos = (np.array(gt_labels) == 1).sum()
    amount_neg = (np.array(gt_labels) == 0).sum()
    # Get TP and FP
    TP, FP = 0, 0
    for pred, gt in zip(pred_labels, gt_labels):
        # If frame was considered as having a lost object
        if pred == 1:
            # Predicted an object and there was an object
            if gt == 1:
                TP += 1
            # Predicted an object but there was no object
            else:
                FP += 1
    # TP_rate: TP / (TP+FN) and TP+FN = the total number of positives
    TP_rate = TP / amount_pos if amount_pos != 0 else 0.
    # FP_rate: FP / (FP+TN) and FP+TN = the total number of negatives
    FP_rate = FP / amount_neg if amount_neg != 0 else 0.
    return {
        'TP_rate': TP_rate,
        'FP_rate': FP_rate,
        'total_gt_pos': amount_pos,
        'total_gt_neg': amount_neg,
        'TP': TP,
        'FP': FP,
    }

import cv2
import numpy as np

from .utils_functions import get_multiplication_mask


def get_positives_negatives(pred_labels, gt_labels):
    # Get amount of positives and negatives
    amount_pos = (np.array(gt_labels) == 1).sum()
    amount_neg = (np.array(gt_labels) == 0).sum()
    # Get TP and FP
    TP, FP, TN, FN = 0, 0, 0, 0
    for pred, gt in zip(pred_labels, gt_labels):
        # If frame was considered as having a lost object (TP, FP)
        if pred == 1:
            # Predicted an object and there was an object
            if gt == 1:
                TP += 1
            # Predicted an object but there was no object
            else:
                FP += 1
        # If frame was considered as not having a lost object (FN or TN)
        elif pred == 0:
            if gt == 0:
                TN += 1
            else:
                FN += 1
    assert TP + FN == amount_pos
    assert TN + FP == amount_neg
    return {
        'groundtruth positives': amount_pos,
        'groundtruth negatives': amount_neg,
        'sum tp': TP,
        'sum fp': FP,
        'sum tn': TN,
        'sum fn': FN
    }


def compute_dis_overall(sum_gt_pos, sum_gt_neg, sum_tp, sum_fp, sum_tn, sum_fn):
    den1 = sum_tp + sum_fn
    den2 = sum_fp + sum_tn
    tpr = sum_tp / den1 if den1 != 0 else 0
    fpr = sum_fp / den2 if den2 != 0 else 0
    tpr2 = 0 if sum_gt_pos == 0 else sum_tp / sum_gt_pos
    assert tpr == tpr2
    fpr2 = 0 if sum_gt_neg == 0 else sum_fp / sum_gt_neg
    assert fpr == fpr2
    return {'TPR': tpr, 'FPR': fpr, 'DIS': np.sqrt((1 - tpr)**2 + fpr**2)}


def compute_DIS_pixel_level(gts_dict, pred_blobs, alignment):
    # First, let's get an image containing the gt bounding box represented by a white area
    gt_masks = get_multiplication_mask(gts_dict, alignment)
    gt_masks = gt_masks.squeeze().cpu().numpy().astype(np.uint8) * 255
    acc_bb, acc_tn, acc_fp, acc_fn, acc_tp = [], [], [], [], []
    for pred, gt in zip(pred_blobs, gt_masks):
        bb, tn, fp, fn, tp = frame_eval_pixel(pred, gt)
        acc_bb.append(bb)
        acc_tn.append(tn)
        acc_fp.append(fp)
        acc_fn.append(fn)
        acc_tp.append(tp)
    # 2 - Contabilização de cada vídeo
    # sum the 201 valores of bb, tn, fp, fn e tp
    bb_sum, tn_sum, fp_sum, fn_sum, tp_sum = sum(acc_bb), sum(acc_tn), sum(acc_fp), sum(
        acc_fn), sum(acc_tp)
    # 3 - Compute rates
    tp_rate = tp_sum / bb_sum
    # computing fp rate as the previous benchmarking works do
    fp_rate = fp_sum / len(acc_tp)
    # 4 - Compute do DIS
    dis = np.sqrt((1 - tp_rate)**2 + fp_rate**2)
    accuracy = (tp_sum + tn_sum) / (tp_sum + tn_sum + fp_sum + fn_sum)
    return {
        'DIS': dis,
        'TPR': tp_rate,
        'FPR': fp_rate,
        'list_bb': acc_bb,
        'list_tn': acc_tn,
        'list_fp': acc_fp,
        'list_fn': acc_fn,
        'list_tp': acc_tp,
        'accuracy': accuracy,
        'sum_tp': tp_sum,
        'sum_fp': fp_sum,
        'sum_tn': tn_sum,
        'sum_fn': fn_sum,
        'groundtruth_pos': bb_sum,
        'groundtruth_neg': sum([1 for i in acc_bb if i == 0])
    }


def frame_eval_pixel(vid, gt):
    # vid - predicted frame
    # gt  - ground truth
    assert vid.shape == gt.shape
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

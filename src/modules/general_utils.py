import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def create_cone(max_radius=20.):
    # Reference: morphology/notebooks/approach_04.ipynb
    step = 1.
    X = torch.arange(-max_radius, max_radius + step, step)
    Y = torch.arange(-max_radius, max_radius + step, step)
    X, Y = torch.meshgrid(X, Y)
    # Define a perfect cone
    Z = torch.sqrt(X**2 + Y**2)
    cone = torch.clamp(1 - (Z / (max_radius)), 0, 1)
    return cone


def pencil_func(x, tan_beta, threshold, min_allowed_thresh=0., max_allowed_thresh=1.):
    output = torch.where(x <= threshold, torch.tensor(0.),
                         (tan_beta * x) + 1 - (max_allowed_thresh * tan_beta))
    return output.clamp_(min_allowed_thresh, max_allowed_thresh)


def platan_func(x, tan_beta, threshold, min_allowed_thresh=0., max_allowed_thresh=1.):
    x = x - threshold
    output = torch.where(x <= 0, tan_beta * (x + threshold),
                         (tan_beta * (x + threshold)) + 1 - (max_allowed_thresh * tan_beta))
    return output.clamp_(min_allowed_thresh, max_allowed_thresh)


# FUNCIONANDO! :)
def hardtanh(x, alpha, thresh):
    x = x + 0.5 - thresh
    return torch.clamp((alpha * x) + (0.5 * (1 - alpha)), 0, 1)


def ada_tanh(alpha, beta, thresh, x):
    input = x.clone()
    return torch.clamp((beta * input) + (1 - beta) * hardtanh(input, alpha, thresh), 0, 1)
    # Eduardo
    # return (1/((2*beta)+1))*(beta+0.5+(.5*torch.nn.functional.hardtanh(alpha*x))+(beta*x))


def adatanh_structuring_element_3D(alpha, beta, radius, max_radius, x):
    input = x.clone()
    # thresh = 1-(1/max_radius)*(radius)
    thresh = 1 - radius / max_radius
    return torch.clamp((beta * x) + (1 - beta) * hardtanh(x, alpha, thresh), 0, 1)


def my_function1(alpha, beta, thresh, x):
    alpha += 1e-10

    res = x.clone()

    a = thresh - (alpha / 2)
    b = thresh + (alpha / 2)
    p = a * beta
    q = (b * beta) + 1 - beta

    # Define regions
    region0 = x.le(0)
    region1 = x.gt(0) & x.lt(a)
    region2 = x.ge(a) & x.le(b)
    region3 = x.gt(b) & x.lt(1)
    region4 = x.ge(1)

    # Apply operations in regions
    # x.where(~region1 , x*2) # se puder usar inplace, economizamos memória
    # x.clamp_(min=0, max=1)
    x = torch.where(region1, x * beta, x)
    x = torch.where(region2, ((p * b) - (q * a) - (x * (p - q))) / (b - a), x)
    x = torch.where(region3, (x * beta) + 1 - beta, x)
    x = torch.where(region0, x * 0, x)
    x = torch.where(region4, torch.ones(x.shape, dtype=x.dtype), x)

    return x


def my_function1(alpha, beta, thresh, x):
    if alpha == 0:
        alpha += 0.0000000001
    elif alpha == 1:
        alpha -= 0.0000000001
    res = x.clone()

    a = thresh - (alpha / 2)
    b = thresh + (alpha / 2)
    p = a * beta
    q = (b * beta) + 1 - beta

    def functions(i):
        # region 0
        if i <= 0:
            return 0
        # region 4
        if 1 <= i:
            return 1
        # region 1
        if 0 <= i <= a:
            return i * beta
        # region 2
        if a <= i <= b:
            return ((p * b) - (q * a) - (i * (p - q))) / (b - a)
        # # region 3
        if b <= i <= 1:
            return (i * beta) + 1 - beta
        else:
            xa = 123

    res.apply_(functions)
    return res


def function_structuring_element(curvature, radius, x, y):
    return 1 / (1 + torch.exp(curvature * (x**2 + y**2 - radius**2)))


def function_2D_structuring_element(curvature, radius, x):
    return 1 / (1 + torch.exp(curvature * (x**2 - radius**2)))


def function_step_function(value, init_step, end_step, step=.1, init_end_plot=None):
    if init_end_plot is None:
        x_range = torch.arange(init_step - 1, end_step + 1 + step, step=step)
    else:
        x_range = torch.arange(init_end_plot[0], init_end_plot[1] + step, step=step)
    if init_step != end_step:
        return x_range, ((init_step <= x_range) & (x_range <= end_step)) * value
    else:
        return x_range, x_range * 0


def function_sigmoid(x, skew=0.5, thresh=0., multiplicador=torch.tensor(1.), mirror=False):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if not torch.is_tensor(skew):
        skew = torch.tensor(skew)
    if not torch.is_tensor(thresh):
        thresh = torch.tensor(thresh)
    if not torch.is_tensor(multiplicador):
        multiplicador = torch.tensor(multiplicador)
    # left side is 1, right side is 0
    if mirror:
        return multiplicador * 1 / (1 + torch.exp(skew * (x - thresh)))
    # left side is 0, right side is 1
    else:
        return multiplicador * 1 / (1 + torch.exp(-skew * (x - thresh)))


def save_img(img, img_name, is_binary=False):
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    img = img.squeeze()
    if len(img.shape) == 2:
        # Make it a 3-channel image
        img = np.stack((img.squeeze(), ) * 3, axis=2)
    if is_binary:
        img = (img * 255).astype(np.uint8)
    cv2.imwrite(img_name, img)


# Define functions
def threshold2radius(threshold, max_radius):
    '''Convert a given threshold [0~1] to radius [0-max_radius]'''
    return -max_radius * (threshold - 1)


def radius2threshold(radius, max_radius):
    '''Convert a given radius [0-max_radius] to threshold [0~1]'''
    return 1 - (radius / max_radius)


def function_change_dynamic_range(x, v_xmin, v_xmax):
    return (x - v_xmin) / (v_xmax - v_xmin)
    out = torch.zeros(x.shape).type(x.type())
    for batch in range(x.shape[0]):
        out[batch] = ((x[batch] - v_xmin) / (v_xmax - v_xmin))
    return out


def function_cross_entropy(gt, pred):
    # Implementacao da cross entropy
    res = 0
    for g, p in zip(gt, pred):
        res += (g * torch.log(p)) + ((1 - g) * torch.log(1 - p))
    return -(res.sum()) / len(pred)


def get_eucl_dist(resnet, layer, ref_frames, tar_frames, weights_ref, weights_tar):
    assert ref_frames.shape == tar_frames.shape
    # Get features with Resnet
    feat_tar = resnet.get_features(tar_frames, layer)
    feat_ref = resnet.get_features(ref_frames, layer)
    # Ponderate ref and tar by their weights
    if feat_ref.dim() == 3:
        feat_ref = feat_ref.unsqueeze(0)
    if feat_tar.dim() == 3:
        feat_tar = feat_tar.unsqueeze(0)
    ref = torch.einsum('bchw,c->bchw', [feat_ref, weights_ref])
    tar = torch.einsum('bchw,c->bchw', [feat_tar, weights_tar])
    # Obtain euclidean distance
    out = ((ref - tar)**2).sum(axis=1)
    return out


def binarize_with_otsu(tensor):
    samples = tensor.shape[0]
    for batch in range(samples):
        # Change dynamic range to 0 ~ 255
        tensor[batch] = 255 * ((tensor[batch] - tensor[batch].min()) /
                               (tensor[batch].max() - tensor[batch].min()))
        # Round values and convert to uint8
        tensor[batch] = tensor[batch].round()
        threshold, binary = cv2.threshold(tensor[batch].cpu().numpy().astype(np.uint8), 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Transform binary to torch and change range back to 0 ~ 1
        tensor[batch] = torch.from_numpy(binary).float() / 255
    tensor = torch.unsqueeze(tensor, 1)
    return tensor


# Get features and binarize the difference of the tensors
def get_eucl_dist_and_binarize_tensors(resnet, layer, ref_frames, tar_frames, weights_ref,
                                       weights_tar, device):
    # Get euclidean distance
    out = get_eucl_dist(resnet, layer, ref_frames, tar_frames, weights_ref, weights_tar)
    # Binarize
    out = binarize_with_otsu(out)
    return out.to(device)
    # assert ref_frames.shape == tar_frames.shape
    # # Get features with Resnet
    # feat_tar = resnet.get_features(tar_frames, layer)
    # feat_ref = resnet.get_features(ref_frames, layer)
    # # Ponderate ref and tar by their weights
    # if feat_ref.dim() == 3:
    #     feat_ref= feat_ref.unsqueeze(0)
    # if feat_tar.dim() == 3:
    #     feat_tar= feat_tar.unsqueeze(0)
    # train_batch_size = feat_ref.shape[0]
    # ref = torch.einsum('bchw,c->bchw', [feat_ref, weights_ref])
    # tar = torch.einsum('bchw,c->bchw', [feat_tar, weights_tar])
    # Obtain euclidean distance
    # out = ((ref - tar)**2).sum(axis=1)
    # Binarize with Otsu (threshold is calculated individually for each channel)
    # for batch in range(train_batch_size):
    #     # Change dynamic range to 0 ~ 255
    #     out[batch] = 255 * ((out[batch] - out[batch].min()) /
    #                         (out[batch].max() - out[batch].min()))
    #     # Round values and convert to uint8
    #     out[batch] = out[batch].round()
    #     threshold, binary = cv2.threshold(out[batch].cpu().numpy().astype(np.uint8), 0, 255,
    #                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     # Transform binary to torch and change range back to 0 ~ 1
    #     out[batch] = torch.from_numpy(binary).float().to(device) / 255
    # out = torch.unsqueeze(out, 1)
    # return out


def create_training_img_from_binarized(img, desired_output_shape):
    resized_img = torch.zeros(desired_output_shape)
    img = (img.squeeze(1) * 255).cpu().numpy().astype(np.uint8)
    for i, batch in enumerate(img):
        resized_img[i] = torch.tensor(
            np.array(
                Image.fromarray(batch).resize(
                    desired_output_shape[2:][::-1]).convert(mode="RGB"))).permute(2, 0, 1)
    return resized_img


def gradients_vanished_or_exploded(gradients):
    # When gradient explodes/vanishes, its value is nan
    for grad in gradients:
        # if grad is None or torch.isnan(torch.tensor(grad)):
        # If any gradient is True
        #if grad is None or torch.isnan(torch.tensor(grad)).sum() != 0:
        if grad is None or torch.isnan(grad).sum() != 0:
            return True
    return False

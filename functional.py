import numbers
import cv2
import numpy as np
import PIL
import torch


def _is_tensor_clip(clip):
    """
    [CF] 内部辅助函数：判断输入是否是一个有效的视频张量片段。
    一个有效的张量片段应该是一个 PyTorch 张量，并且维度为 4。
    预期的 4 个维度是: (C, T, H, W) -> (通道, 时间, 高度, 宽度)
    """
    return torch.is_tensor(clip) and clip.ndimension() == 4


def crop_clip(clip, min_h, min_w, h, w):
    """
    [CF] 对视频片段的每一帧进行"相同位置"的裁剪。
    
    这是视频增强的关键操作：必须确保所有帧的裁剪区域完全一致，
    这样才能保持视频的时空连续性，避免引入人为的抖动。

    Args:
        clip (list): 一个包含多帧的列表，元素可以是 PIL.Image 或 np.ndarray。
        min_h, min_w (int): 裁剪起点的 y, x 坐标。
        h, w (int): 裁剪区域的高度和宽度。

    Returns:
        list: 裁剪后的图像列表，格式与输入相同。
    """
    if isinstance(clip[0], np.ndarray):
        # [CF] 如果是 numpy 数组，直接使用数组切片进行裁剪。
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]

    elif isinstance(clip[0], PIL.Image.Image):
        # [CF] 如果是 PIL 图像，则调用其 .crop() 方法。
        # .crop() 的参数是一个元组: (left, top, right, bottom)
        cropped = [
            img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip
        ]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return cropped


def resize_clip(clip, size, interpolation='bilinear'):
    """
    [CF] 对视频片段的每一帧进行缩放。
    支持两种缩放方式：
    1. 指定目标短边长度 (size 为数字)，保持长宽比缩放。
    2. 指定目标精确尺寸 (size 为 (宽, 高) 元组)。

    Args:
        clip (list): 图像列表 (PIL.Image 或 np.ndarray)。
        size (int or tuple): 目标尺寸。
        interpolation (str): 插值方法，'bilinear' 或 'nearest'。

    Returns:
        list: 缩放后的图像列表。
    """
    if isinstance(clip[0], np.ndarray):
        if isinstance(size, numbers.Number):
            im_h, im_w, im_c = clip[0].shape
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[0], size[1]
        if interpolation == 'bilinear':
            np_inter = cv2.INTER_LINEAR
        else:
            np_inter = cv2.INTER_NEAREST
        scaled = [
            cv2.resize(img, size, interpolation=np_inter) for img in clip
        ]
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        # [CF] 根据插值方法选择 OpenCV 的插值标志
        if interpolation == 'bilinear':
            pil_inter = PIL.Image.BILINEAR
        else:
            pil_inter = PIL.Image.NEAREST
        # [CF] 使用 OpenCV 逐帧缩放
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return scaled


def get_resize_sizes(im_h, im_w, size):
    """
    [CF] 辅助函数：计算保持长宽比的缩放后尺寸。
    
    给定原始高宽和一个目标短边长度，计算出缩放后的 (高, 宽)，
    确保原始图像的短边被缩放到 size，长边按相同比例缩放。

    Args:
        im_h, im_w (int): 原始图像的高和宽。
        size (int): 目标短边长度。

    Returns:
        tuple: (new_height, new_width)
    """
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow


def normalize(clip, mean, std, inplace=False):
    """
    [CF] 对"张量格式"的视频片段进行标准化: (clip - mean) / std。
    
    注意：此函数要求输入的 clip 已经是 PyTorch 张量，
    并且形状为 (C, T, H, W)。

    Args:
        clip (Tensor): 形状为 (C, T, H, W) 的视频张量。
        mean (tuple/list): 各通道的均值，长度等于 C。
        std (tuple/list): 各通道的标准差，长度等于 C。
        inplace (bool): 是否原地修改张量以节省内存。

    Returns:
        Tensor: 标准化后的视频张量。
    """
    if not _is_tensor_clip(clip):
        raise TypeError('tensor is not a torch clip.')

    if not inplace:
        # [CF] 如果不原地操作，则克隆一份，避免影响原始数据。
        clip = clip.clone()

    dtype = clip.dtype
    # [CF] 将 mean 和 std 转换为与 clip 相同类型和设备的张量。
    # 并增加维度，使其形状变为 [C, 1, 1, 1]，以便广播到 [C, T, H, W]。
    mean = torch.as_tensor(mean, dtype=dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=dtype, device=clip.device)
    # [CF] 执行标准化操作。sub_ 和 div_ 是原地操作版本。
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])

    return clip

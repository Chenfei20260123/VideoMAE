#!/usr/bin/env python3
# [CF] 2026-04-12:
# 这个文件定义了视频数据增强的各种变换操作。
# 与 transforms.py 中的 "Group" 类不同，这里的函数直接操作张量格式的视频，
# 并支持同步处理边界框（boxes），适用于视频分类和目标检测任务。

#!/usr/bin/env python3
import math
import numpy as np
import random
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms

from rand_augment import rand_augment_transform
from random_erasing import RandomErasing


import numbers
import PIL
import torchvision

import functional as FF

# PIL 插值方法到字符串的映射（用于调试和日志）
_pil_interpolation_to_str = {
    Image.NEAREST: "PIL.Image.NEAREST",
    Image.BILINEAR: "PIL.Image.BILINEAR",
    Image.BICUBIC: "PIL.Image.BICUBIC",
    Image.LANCZOS: "PIL.Image.LANCZOS",
    Image.HAMMING: "PIL.Image.HAMMING",
    Image.BOX: "PIL.Image.BOX",
}

# 随机插值方法的选择范围
_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


def _pil_interp(method):
    if method == "bicubic":
        return Image.BICUBIC
    elif method == "lanczos":
        return Image.LANCZOS
    elif method == "hamming":
        return Image.HAMMING
    else:
        return Image.BILINEAR


def random_short_side_scale_jitter(
    images, min_size, max_size, boxes=None, inverse_uniform_sampling=False
):
    """
    Perform a spatial short scale jittering on the given images and
    corresponding boxes.
    Args:
        images (tensor): images to perform scale jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
        min_size (int): the minimal size to scale the frames.
        max_size (int): the maximal size to scale the frames.
        boxes (ndarray): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale, max_scale].
    Returns:
        (tensor): the scaled images with dimension of
            `num frames` x `channel` x `new height` x `new width`.
        (ndarray or None): the scaled boxes with dimension of
            `num boxes` x 4.
    """    
    """
    [CF] 随机短边缩放抖动。
    
    对视频的每一帧进行保持长宽比的缩放，缩放后的短边长度在 [min_size, max_size] 范围内随机选择。
    这是视频数据增强中最基础的空间变换之一。
    
    Args:
        images (tensor): 输入视频，形状为 (T, C, H, W)
        min_size (int): 短边的最小尺寸
        max_size (int): 短边的最大尺寸
        boxes (ndarray, optional): 边界框，形状为 (N, 4)，会同步缩放
        inverse_uniform_sampling (bool): 是否使用倒数均匀采样
    
    Returns:
        images (tensor): 缩放后的视频
        boxes (ndarray or None): 缩放后的边界框
    """
    if inverse_uniform_sampling:
        # 在 [1/max_size, 1/min_size] 范围内均匀采样，然后取倒数
        size = int(
            round(1.0 / np.random.uniform(1.0 / max_size, 1.0 / min_size))
        )
    else:
        # 直接在 [min_size, max_size] 范围内均匀采样
        size = int(round(np.random.uniform(min_size, max_size)))

    height = images.shape[2]
    width = images.shape[3]
    # 如果短边已经等于目标尺寸，直接返回
    if (width <= height and width == size) or (
        height <= width and height == size
    ):
        return images, boxes
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
        if boxes is not None:
            boxes = boxes * float(new_height) / height
    else:
        new_width = int(math.floor((float(width) / height) * size))
        if boxes is not None:
            boxes = boxes * float(new_width) / width

    return (
        torch.nn.functional.interpolate(
            images,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        ),
        boxes,
    )


def crop_boxes(boxes, x_offset, y_offset):
    """
    Peform crop on the bounding boxes given the offsets.
    Args:
        boxes (ndarray or None): bounding boxes to peform crop. The dimension
            is `num boxes` x 4.
        x_offset (int): cropping offset in the x axis.
        y_offset (int): cropping offset in the y axis.
    Returns:
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """    
    """
    [CF] 根据裁剪偏移调整边界框坐标。
    
    Args:
        boxes (ndarray): 边界框，形状为 (N, 4)，格式为 [x1, y1, x2, y2]
        x_offset (int): X 轴方向的裁剪偏移
        y_offset (int): Y 轴方向的裁剪偏移
    
    Returns:
        cropped_boxes (ndarray): 调整后的边界框
    """
    cropped_boxes = boxes.copy()
    cropped_boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
    cropped_boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset

    return cropped_boxes


def random_crop(images, size, boxes=None):
    """
    Perform random spatial crop on the given images and corresponding boxes.
    Args:
        images (tensor): images to perform random crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): the size of height and width to crop on the image.
        boxes (ndarray or None): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
    Returns:
        cropped (tensor): cropped images with dimension of
            `num frames` x `channel` x `size` x `size`.
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """    
    """
    [CF] 随机空间裁剪。
    
    在视频的每一帧上执行相同位置的随机裁剪，保持时空一致性。
    
    Args:
        images (tensor): 输入视频，形状为 (T, C, H, W)
        size (int): 裁剪后的目标尺寸（正方形）
        boxes (ndarray, optional): 边界框，会同步调整
    
    Returns:
        cropped (tensor): 裁剪后的视频，形状为 (T, C, size, size)
        cropped_boxes (ndarray or None): 调整后的边界框
    """
    if images.shape[2] == size and images.shape[3] == size:
        return images
    height = images.shape[2]
    width = images.shape[3]
    # 随机生成裁剪起点
    y_offset = 0
    if height > size:
        y_offset = int(np.random.randint(0, height - size))
    x_offset = 0
    if width > size:
        x_offset = int(np.random.randint(0, width - size))
    # 对所有帧应用相同的裁剪
    cropped = images[
        :, :, y_offset : y_offset + size, x_offset : x_offset + size
    ]

    cropped_boxes = (
        crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None
    )

    return cropped, cropped_boxes


def horizontal_flip(prob, images, boxes=None):
    """
    Perform horizontal flip on the given images and corresponding boxes.
    Args:
        prob (float): probility to flip the images.
        images (tensor): images to perform horizontal flip, the dimension is
            `num frames` x `channel` x `height` x `width`.
        boxes (ndarray or None): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
    Returns:
        images (tensor): images with dimension of
            `num frames` x `channel` x `height` x `width`.
        flipped_boxes (ndarray or None): the flipped boxes with dimension of
            `num boxes` x 4.
    """
    """
    [CF] 随机水平翻转。
    
    以概率 prob 对视频的每一帧进行水平翻转。对于视频分类任务，
    翻转可能会改变动作方向，因此对于 SSV2 等方向敏感的数据集，通常禁用此操作。
    
    Args:
        prob (float): 翻转概率
        images (tensor): 输入视频，形状为 (T, C, H, W)
        boxes (ndarray, optional): 边界框
    
    Returns:
        images (tensor): 翻转后的视频
        flipped_boxes (ndarray or None): 翻转后的边界框
    """
    if boxes is None:
        flipped_boxes = None
    else:
        flipped_boxes = boxes.copy()

    if np.random.uniform() < prob:
        images = images.flip((-1))

        if len(images.shape) == 3:
            width = images.shape[2]
        elif len(images.shape) == 4:
            width = images.shape[3]
        else:
            raise NotImplementedError("Dimension does not supported")
        if boxes is not None:
            flipped_boxes[:, [0, 2]] = width - boxes[:, [2, 0]] - 1

    return images, flipped_boxes


def uniform_crop(images, size, spatial_idx, boxes=None, scale_size=None):
    """
    Perform uniform spatial sampling on the images and corresponding boxes.
    Args:
        images (tensor): images to perform uniform crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): size of height and weight to crop the images.
        spatial_idx (int): 0, 1, or 2 for left, center, and right crop if width
            is larger than height. Or 0, 1, or 2 for top, center, and bottom
            crop if height is larger than width.
        boxes (ndarray or None): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
        scale_size (int): optinal. If not None, resize the images to scale_size before
            performing any crop.
    Returns:
        cropped (tensor): images with dimension of
            `num frames` x `channel` x `size` x `size`.
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """    
    """
    [CF] 均匀空间裁剪（用于测试阶段）。
    
    在测试时，为了获得确定性的结果，通常采用固定的裁剪方式：
    - spatial_idx=0: 左/上裁剪
    - spatial_idx=1: 中心裁剪
    - spatial_idx=2: 右/下裁剪
    
    Args:
        images (tensor): 输入视频，形状为 (T, C, H, W)
        size (int): 裁剪尺寸
        spatial_idx (int): 裁剪位置索引（0, 1, 2）
        boxes (ndarray, optional): 边界框
        scale_size (int, optional): 裁剪前先将短边缩放到此尺寸
    
    Returns:
        cropped (tensor): 裁剪后的视频
        cropped_boxes (ndarray or None): 调整后的边界框
    """
    assert spatial_idx in [0, 1, 2]
    ndim = len(images.shape)
    if ndim == 3:
        images = images.unsqueeze(0)
    height = images.shape[2]
    width = images.shape[3]

    if scale_size is not None:
        if width <= height:
            width, height = scale_size, int(height / width * scale_size)
        else:
            width, height = int(width / height * scale_size), scale_size
        images = torch.nn.functional.interpolate(
            images,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = images[
        :, :, y_offset : y_offset + size, x_offset : x_offset + size
    ]
    cropped_boxes = (
        crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None
    )
    if ndim == 3:
        cropped = cropped.squeeze(0)
    return cropped, cropped_boxes


def clip_boxes_to_image(boxes, height, width):
    """
    Clip an array of boxes to an image with the given height and width.
    Args:
        boxes (ndarray): bounding boxes to perform clipping.
            Dimension is `num boxes` x 4.
        height (int): given image height.
        width (int): given image width.
    Returns:
        clipped_boxes (ndarray): the clipped boxes with dimension of
            `num boxes` x 4.
    """    
    """
    [CF] 将边界框裁剪到图像范围内。
    
    确保边界框坐标不超出图像边界。
    """
    clipped_boxes = boxes.copy()
    clipped_boxes[:, [0, 2]] = np.minimum(
        width - 1.0, np.maximum(0.0, boxes[:, [0, 2]])
    )
    clipped_boxes[:, [1, 3]] = np.minimum(
        height - 1.0, np.maximum(0.0, boxes[:, [1, 3]])
    )
    return clipped_boxes


def blend(images1, images2, alpha):
    """
    Blend two images with a given weight alpha.
    Args:
        images1 (tensor): the first images to be blended, the dimension is
            `num frames` x `channel` x `height` x `width`.
        images2 (tensor): the second images to be blended, the dimension is
            `num frames` x `channel` x `height` x `width`.
        alpha (float): the blending weight.
    Returns:
        (tensor): blended images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """    
    """
    [CF] 图像混合（用于 Mixup 等增强策略）。
    
    将两个视频按比例混合：output = images1 * alpha + images2 * (1 - alpha)
    """
    return images1 * alpha + images2 * (1 - alpha)


def grayscale(images):
    """
    Get the grayscale for the input images. The channels of images should be
    in order BGR.
    Args:
        images (tensor): the input images for getting grayscale. Dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        img_gray (tensor): blended images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """    
    """
    [CF] 转换为灰度图。
    
    注意：输入图像的通道顺序应为 BGR（OpenCV 格式）。
    公式：Gray = 0.299*R + 0.587*G + 0.114*B
    """
    # R -> 0.299, G -> 0.587, B -> 0.114.
    img_gray = torch.tensor(images)
    gray_channel = (
        0.299 * images[:, 2] + 0.587 * images[:, 1] + 0.114 * images[:, 0]
    )
    img_gray[:, 0] = gray_channel
    img_gray[:, 1] = gray_channel
    img_gray[:, 2] = gray_channel
    return img_gray


def color_jitter(images, img_brightness=0, img_contrast=0, img_saturation=0):
    """
    Perfrom a color jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
        img_brightness (float): jitter ratio for brightness.
        img_contrast (float): jitter ratio for contrast.
        img_saturation (float): jitter ratio for saturation.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """    
    """
    [CF] 颜色抖动。
    
    随机调整视频的亮度、对比度和饱和度。三种操作的应用顺序是随机的，
    以增加数据增强的多样性。
    
    Args:
        images (tensor): 输入视频，通道顺序为 BGR
        img_brightness (float): 亮度调整幅度
        img_contrast (float): 对比度调整幅度
        img_saturation (float): 饱和度调整幅度
    
    Returns:
        images (tensor): 颜色抖动后的视频
    """

    jitter = []
    if img_brightness != 0:
        jitter.append("brightness")
    if img_contrast != 0:
        jitter.append("contrast")
    if img_saturation != 0:
        jitter.append("saturation")

    if len(jitter) > 0:
        order = np.random.permutation(np.arange(len(jitter)))
        for idx in range(0, len(jitter)):
            if jitter[order[idx]] == "brightness":
                images = brightness_jitter(img_brightness, images)
            elif jitter[order[idx]] == "contrast":
                images = contrast_jitter(img_contrast, images)
            elif jitter[order[idx]] == "saturation":
                images = saturation_jitter(img_saturation, images)
    return images


def brightness_jitter(var, images):
    """
    Perfrom brightness jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        var (float): jitter ratio for brightness.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """    
    """
    [CF] 亮度抖动。
    
    通过将图像与全零图像（黑色）混合来调整亮度。
    alpha > 1 时变亮，alpha < 1 时变暗。
    
    Args:
        var (float): 抖动幅度，alpha 在 [1-var, 1+var] 范围内随机采样
        images (tensor): 输入视频，形状为 (T, C, H, W)
    
    Returns:
        images (tensor): 亮度调整后的视频
    """
    alpha = 1.0 + np.random.uniform(-var, var)

    img_bright = torch.zeros(images.shape)
    images = blend(images, img_bright, alpha)
    return images


def contrast_jitter(var, images):
    """
    Perfrom contrast jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        var (float): jitter ratio for contrast.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    """
    [CF] 对比度抖动。
    
    通过将图像与其灰度图的全局平均值混合来调整对比度。
    混合比例为 alpha 时，对比度改变。
    
    Args:
        var (float): 抖动幅度
        images (tensor): 输入视频，形状为 (T, C, H, W)，通道顺序为 BGR
    
    Returns:
        images (tensor): 对比度调整后的视频
    """
    alpha = 1.0 + np.random.uniform(-var, var)

    img_gray = grayscale(images)
    img_gray[:] = torch.mean(img_gray, dim=(1, 2, 3), keepdim=True)
    images = blend(images, img_gray, alpha)
    return images


def saturation_jitter(var, images):
    """
    Perfrom saturation jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        var (float): jitter ratio for saturation.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """    
    """
    [CF] 饱和度抖动。
    
    通过将图像与其灰度版本混合来调整饱和度。
    alpha > 1 时饱和度增加，alpha < 1 时趋向灰度。
    
    Args:
        var (float): 抖动幅度
        images (tensor): 输入视频，形状为 (T, C, H, W)，通道顺序为 BGR
    
    Returns:
        images (tensor): 饱和度调整后的视频
    """
    alpha = 1.0 + np.random.uniform(-var, var)
    img_gray = grayscale(images)
    images = blend(images, img_gray, alpha)

    return images


def lighting_jitter(images, alphastd, eigval, eigvec):
    """
    Perform AlexNet-style PCA jitter on the given images.
    Args:
        images (tensor): images to perform lighting jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
        alphastd (float): jitter ratio for PCA jitter.
        eigval (list): eigenvalues for PCA jitter.
        eigvec (list[list]): eigenvectors for PCA jitter.
    Returns:
        out_images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """    
    """
    [CF] AlexNet 风格的 PCA 颜色增强（光照抖动）。
    
    通过对 RGB 通道的主成分进行扰动，模拟自然光照变化。
    这是一种经典的数据增强技术，在 ImageNet 训练中被广泛使用。
    
    原理：在 RGB 颜色的 PCA 主成分方向上添加噪声。
    
    Args:
        images (tensor): 输入视频，形状为 (T, C, H, W) 或 (C, H, W)
        alphastd (float): 噪声的标准差
        eigval (list): PCA 特征值（3个）
        eigvec (list[list]): PCA 特征向量（3x3矩阵）
    
    Returns:
        out_images (tensor): 颜色增强后的视频
    """
    if alphastd == 0:
        return images
    # generate alpha1, alpha2, alpha3.
    alpha = np.random.normal(0, alphastd, size=(1, 3))
    eig_vec = np.array(eigvec)
    eig_val = np.reshape(eigval, (1, 3))
    rgb = np.sum(
        eig_vec * np.repeat(alpha, 3, axis=0) * np.repeat(eig_val, 3, axis=0),
        axis=1,
    )
    out_images = torch.zeros_like(images)
    if len(images.shape) == 3:
        # C H W
        channel_dim = 0
    elif len(images.shape) == 4:
        # T C H W
        channel_dim = 1
    else:
        raise NotImplementedError(f"Unsupported dimension {len(images.shape)}")

    for idx in range(images.shape[channel_dim]):
        # C H W
        if len(images.shape) == 3:
            out_images[idx] = images[idx] + rgb[2 - idx]
        # T C H W
        elif len(images.shape) == 4:
            out_images[:, idx] = images[:, idx] + rgb[2 - idx]
        else:
            raise NotImplementedError(
                f"Unsupported dimension {len(images.shape)}"
            )

    return out_images


def color_normalization(images, mean, stddev):
    """
    Perform color nomration on the given images.
    Args:
        images (tensor): images to perform color normalization. Dimension is
            `num frames` x `channel` x `height` x `width`.
        mean (list): mean values for normalization.
        stddev (list): standard deviations for normalization.

    Returns:
        out_images (tensor): the noramlized images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """    
    """
    [CF] 颜色归一化。
    
    对视频的每个通道执行 (x - mean) / std 的标准化操作。
    
    Args:
        images (tensor): 输入视频
        mean (list): 各通道均值
        stddev (list): 各通道标准差
    
    Returns:
        out_images (tensor): 归一化后的视频
    """
    if len(images.shape) == 3:
        assert (
            len(mean) == images.shape[0]
        ), "channel mean not computed properly"
        assert (
            len(stddev) == images.shape[0]
        ), "channel stddev not computed properly"
    elif len(images.shape) == 4:
        assert (
            len(mean) == images.shape[1]
        ), "channel mean not computed properly"
        assert (
            len(stddev) == images.shape[1]
        ), "channel stddev not computed properly"
    else:
        raise NotImplementedError(f"Unsupported dimension {len(images.shape)}")

    out_images = torch.zeros_like(images)
    for idx in range(len(mean)):
        # C H W
        if len(images.shape) == 3:
            out_images[idx] = (images[idx] - mean[idx]) / stddev[idx]
        elif len(images.shape) == 4:
            out_images[:, idx] = (images[:, idx] - mean[idx]) / stddev[idx]
        else:
            raise NotImplementedError(
                f"Unsupported dimension {len(images.shape)}"
            )
    return out_images


def _get_param_spatial_crop(
    scale, ratio, height, width, num_repeat=10, log_scale=True, switch_hw=False
):
    """
    Given scale, ratio, height and width, return sampled coordinates of the videos.
    """
    """
    [CF] 为随机裁剪生成参数（内部函数）。
    
    在给定的面积比例（scale）和宽高比（ratio）范围内，
    随机采样一个裁剪区域的尺寸和位置。
    
    Args:
        scale (tuple): 裁剪面积相对于原图的比例范围，如 (0.08, 1.0)
        ratio (tuple): 裁剪区域的宽高比范围，如 (3/4, 4/3)
        height, width: 原图的高和宽
        num_repeat (int): 最大尝试次数
        log_scale (bool): 是否在对数空间均匀采样宽高比
        switch_hw (bool): 是否以 50% 概率交换高宽
    
    Returns:
        tuple: (i, j, h, w) 裁剪起点的 y, x 坐标和裁剪区域的高、宽
    """
    for _ in range(num_repeat):
        area = height * width
        target_area = random.uniform(*scale) * area
        if log_scale:
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))
        else:
            aspect_ratio = random.uniform(*ratio)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if np.random.uniform() < 0.5 and switch_hw:
            w, h = h, w

        if 0 < w <= width and 0 < h <= height:
            i = random.randint(0, height - h)
            j = random.randint(0, width - w)
            return i, j, h, w

    # Fallback to central crop
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:  # whole image
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w


def random_resized_crop(
    images,
    target_height,
    target_width,
    scale=(0.8, 1.0),
    ratio=(3.0 / 4.0, 4.0 / 3.0),
):
    """
    Crop the given images to random size and aspect ratio. A crop of random
    size (default: of 0.08 to 1.0) of the original size and a random aspect
    ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This
    crop is finally resized to given size. This is popularly used to train the
    Inception networks.

    Args:
        images: Images to perform resizing and cropping.
        target_height: Desired height after cropping.
        target_width: Desired width after cropping.
        scale: Scale range of Inception-style area based random resizing.
        ratio: Aspect ratio range of Inception-style area based random resizing.
    """
    """
    [CF] 随机缩放裁剪（Inception 风格）。
    
    这是训练 Inception 网络时流行的数据增强方法：
    1. 随机选择裁剪区域的面积和宽高比
    2. 裁剪该区域
    3. 缩放到目标尺寸
    
    注意：视频的所有帧共享相同的裁剪参数，保持时空一致性。
    
    Args:
        images (tensor): 输入视频，形状为 (T, C, H, W)
        target_height, target_width: 目标尺寸
        scale (tuple): 面积比例范围
        ratio (tuple): 宽高比范围
    
    Returns:
        tensor: 裁剪并缩放后的视频，形状为 (T, C, target_height, target_width)
    """

    height = images.shape[2]
    width = images.shape[3]

    i, j, h, w = _get_param_spatial_crop(scale, ratio, height, width)
    cropped = images[:, :, i : i + h, j : j + w]
    return torch.nn.functional.interpolate(
        cropped,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )


def random_resized_crop_with_shift(
    images,
    target_height,
    target_width,
    scale=(0.8, 1.0),
    ratio=(3.0 / 4.0, 4.0 / 3.0),
):
    """
    This is similar to random_resized_crop. However, it samples two different
    boxes (for cropping) for the first and last frame. It then linearly
    interpolates the two boxes for other frames.

    Args:
        images: Images to perform resizing and cropping.
        target_height: Desired height after cropping.
        target_width: Desired width after cropping.
        scale: Scale range of Inception-style area based random resizing.
        ratio: Aspect ratio range of Inception-style area based random resizing.
    """    
    """
    [CF] 带位移的随机缩放裁剪（用于模拟相机运动）。
    
    与 random_resized_crop 不同，此函数为第一帧和最后一帧采样两个不同的裁剪区域，
    中间帧的裁剪区域通过线性插值得到。这模拟了相机缓慢移动或变焦的效果，
    是视频数据增强的高级技巧。
    
    Args:
        images (tensor): 输入视频，形状为 (C, T, H, W)
        target_height, target_width: 目标尺寸
        scale (tuple): 面积比例范围
        ratio (tuple): 宽高比范围
    
    Returns:
        tensor: 裁剪后的视频，形状为 (C, T, target_height, target_width)
    """
    t = images.shape[1]
    height = images.shape[2]
    width = images.shape[3]

    i, j, h, w = _get_param_spatial_crop(scale, ratio, height, width)
    i_, j_, h_, w_ = _get_param_spatial_crop(scale, ratio, height, width)
    i_s = [int(i) for i in torch.linspace(i, i_, steps=t).tolist()]
    j_s = [int(i) for i in torch.linspace(j, j_, steps=t).tolist()]
    h_s = [int(i) for i in torch.linspace(h, h_, steps=t).tolist()]
    w_s = [int(i) for i in torch.linspace(w, w_, steps=t).tolist()]
    out = torch.zeros((3, t, target_height, target_width))
    for ind in range(t):
        out[:, ind : ind + 1, :, :] = torch.nn.functional.interpolate(
            images[
                :,
                ind : ind + 1,
                i_s[ind] : i_s[ind] + h_s[ind],
                j_s[ind] : j_s[ind] + w_s[ind],
            ],
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
    return out


def create_random_augment(
    input_size,
    auto_augment=None,
    interpolation="bilinear",
):
    """
    Get video randaug transform.

    Args:
        input_size: The size of the input video in tuple.
        auto_augment: Parameters for randaug. An example:
            "rand-m7-n4-mstd0.5-inc1" (m is the magnitude and n is the number
            of operations to apply).
        interpolation: Interpolation method.
    """    
    """
    [CF] 创建 RandAugment 数据增强变换。
    
    根据配置字符串创建 RandAugment 变换，用于微调阶段的自动数据增强。
    
    Args:
        input_size: 输入尺寸
        auto_augment (str): RandAugment 配置字符串，如 "rand-m7-n4-mstd0.5-inc1"
            - m: magnitude（强度）
            - n: num_layers（操作数量）
            - mstd: magnitude 噪声标准差
            - inc: 是否使用递增版本的操作
        interpolation: 插值方法
    
    Returns:
        transforms.Compose: RandAugment 变换
    """
    if isinstance(input_size, tuple):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, tuple):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = {"translate_const": int(img_size_min * 0.45)}
        if interpolation and interpolation != "random":
            aa_params["interpolation"] = _pil_interp(interpolation)
        if auto_augment.startswith("rand"):
            return transforms.Compose(
                [rand_augment_transform(auto_augment, aa_params)]
            )
    raise NotImplementedError


def random_sized_crop_img(
    im,
    size,
    jitter_scale=(0.08, 1.0),
    jitter_aspect=(3.0 / 4.0, 4.0 / 3.0),
    max_iter=10,
):
    """
    Performs Inception-style cropping (used for training).
    """    
    """
    [CF] 单张图像的随机缩放裁剪。
    
    与 random_resized_crop 类似，但专门用于单张图像（形状为 (C, H, W)）。
    """
    assert (
        len(im.shape) == 3
    ), "Currently only support image for random_sized_crop"
    h, w = im.shape[1:3]
    i, j, h, w = _get_param_spatial_crop(
        scale=jitter_scale,
        ratio=jitter_aspect,
        height=h,
        width=w,
        num_repeat=max_iter,
        log_scale=False,
        switch_hw=True,
    )
    cropped = im[:, i : i + h, j : j + w]
    return torch.nn.functional.interpolate(
        cropped.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


# The following code are modified based on timm lib, we will replace the following
# contents with dependency from PyTorchVideo.
# https://github.com/facebookresearch/pytorchvideo
class RandomResizedCropAndInterpolation:
    """Crop the given PIL Image to random size and aspect ratio with random interpolation.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """    
    """
    [CF] 随机缩放裁剪 + 随机插值方法。
    
    与 PyTorch 官方的 RandomResizedCrop 类似，但额外支持：
    1. 随机选择插值方法（从 _RANDOM_INTERPOLATION 中选）
    2. 更灵活的尺寸和比例配置
    
    这是训练 Inception 网络时流行的数据增强方法。
    """

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation="bilinear",
    ):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            print("range should be of kind (min, max)")

        if interpolation == "random":
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """        
        """
        [CF] 为随机裁剪生成参数。
        
        在给定面积比例和宽高比范围内，随机采样裁剪区域。
        最多尝试 10 次，失败则回退到中心裁剪。
        
        Args:
            img (PIL Image): 输入图像
            scale (tuple): 面积比例范围
            ratio (tuple): 宽高比范围
        
        Returns:
            tuple: (i, j, h, w) 裁剪起点的 y, x 坐标和裁剪区域的高、宽
        """
        area = img.size[0] * img.size[1]

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
        Returns:
            PIL Image: Randomly cropped and resized image.
        """        
        """
        [CF] 对输入图像应用随机缩放裁剪。
        
        注意：此操作只处理单张 PIL 图像。对于视频，需要在外部循环中对每帧调用，
        并确保所有帧使用相同的裁剪参数以保持时空一致性。
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        return F.resized_crop(img, i, j, h, w, self.size, interpolation)

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = " ".join(
                [_pil_interpolation_to_str[x] for x in self.interpolation]
            )
        else:
            interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + "(size={0}".format(self.size)
        format_string += ", scale={0}".format(
            tuple(round(s, 4) for s in self.scale)
        )
        format_string += ", ratio={0}".format(
            tuple(round(r, 4) for r in self.ratio)
        )
        format_string += ", interpolation={0})".format(interpolate_str)
        return format_string


def transforms_imagenet_train(
    img_size=224,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.0,
    color_jitter=0.4,
    auto_augment=None,
    interpolation="random",
    use_prefetcher=False,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    re_prob=0.0,
    re_mode="const",
    re_count=1,
    re_num_splits=0,
    separate=False,
):
    """
    If separate==True, the transforms are returned as a tuple of 3 separate transforms
    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'clean' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """
        """
    [CF] 创建 ImageNet 风格的训练数据增强流水线。
    
    这是 VideoMAE 微调阶段最常用的数据增强配置函数。它组装了：
    1. 主变换（primary）：缩放裁剪、翻转
    2. 次级变换（secondary）：颜色抖动 或 RandAugment
    3. 最终变换（final）：转张量、归一化、随机擦除
    
    Args:
        img_size (int or tuple): 目标图像尺寸
        scale (tuple): 随机裁剪的面积比例范围，默认 (0.08, 1.0)
        ratio (tuple): 随机裁剪的宽高比范围，默认 (3/4, 4/3)
        hflip (float): 水平翻转概率
        vflip (float): 垂直翻转概率（视频通常为 0）
        color_jitter (float or tuple): 颜色抖动幅度
        auto_augment (str): RandAugment 配置字符串，如 "rand-m7-n4"
        interpolation (str): 插值方法
        mean, std (tuple): 归一化的均值和标准差
        re_prob (float): 随机擦除概率
        re_mode (str): 随机擦除模式
        re_count (int): 随机擦除块数量
        separate (bool): 是否返回分离的变换（用于混合数据集）
    
    Returns:
        transforms.Compose 或 tuple: 组合后的变换
    """
    if isinstance(img_size, tuple):
        img_size = img_size[-2:]
    else:
        img_size = img_size

    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(
        ratio or (3.0 / 4.0, 4.0 / 3.0)
    )  # default imagenet ratio range

    # =========================================================================
    # 主变换：空间几何变换
    # =========================================================================
    primary_tfl = [
        RandomResizedCropAndInterpolation(
            img_size, scale=scale, ratio=ratio, interpolation=interpolation
        )
    ]
    if hflip > 0.0:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.0:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]

    # =========================================================================
    # 次级变换：颜色/外观变换
    # =========================================================================
    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, tuple):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != "random":
            aa_params["interpolation"] = _pil_interp(interpolation)
        if auto_augment.startswith("rand"):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith("augmix"):
            raise NotImplementedError("Augmix not implemented")
        else:
            raise NotImplementedError("Auto aug not implemented")
    elif color_jitter is not None:
        # color jitter is enabled when not using AA
        if isinstance(color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter),) * 3
        secondary_tfl += [transforms.ColorJitter(*color_jitter)]

    # =========================================================================
    # 最终变换：张量转换 + 归一化 + 随机擦除
    # =========================================================================
    final_tfl = []
    final_tfl += [
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
    ]
    if re_prob > 0.0:
        final_tfl.append(
            RandomErasing(
                re_prob,
                mode=re_mode,
                max_count=re_count,
                num_splits=re_num_splits,
                device="cpu",
                cube=False,
            )
        )

    if separate:
        return (
            transforms.Compose(primary_tfl),
            transforms.Compose(secondary_tfl),
            transforms.Compose(final_tfl),
        )
    else:
        return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)

############################################################################################################
############################################################################################################

class Compose(object):
    """Composes several transforms
    Args:
    transforms (list of ``Transform`` objects): list of transforms
    to compose
    """    
    """
    [CF] 变换组合器。
    
    将多个变换串联起来，依次应用到输入的视频片段上。
    这是构建数据处理流水线的标准方式。
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip


class RandomHorizontalFlip(object):
    """Horizontally flip the list of given images randomly
    with a probability 0.5
    """    
    """
    [CF] 随机水平翻转。
    
    以 50% 的概率对视频片段的所有帧进行水平翻转。
    注意：所有帧共享相同的翻转决策，保持时空一致性。
    """

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Randomly flipped clip
        """
        if random.random() < 0.5:
            if isinstance(clip[0], np.ndarray):
                return [np.fliplr(img) for img in clip]
            elif isinstance(clip[0], PIL.Image.Image):
                return [
                    img.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img in clip
                ]
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                ' but got list of {0}'.format(type(clip[0])))
        return clip


class RandomResize(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size
    The larger the original image is, the more times it takes to
    interpolate
    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
    """    
    """
    [CF] 随机缩放。
    
    在给定的宽高比范围内随机选择一个缩放因子，对视频的所有帧进行缩放。
    所有帧共享相同的缩放因子。
    
    Args:
        ratio (tuple): 缩放因子的范围，如 (3/4, 4/3)
        interpolation (str): 插值方法，'nearest' 或 'bilinear'
    """

    def __init__(self, ratio=(3. / 4., 4. / 3.), interpolation='nearest'):
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, clip):
        scaling_factor = random.uniform(self.ratio[0], self.ratio[1])

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_w, new_h)
        resized = FF.resize_clip(
            clip, new_size, interpolation=self.interpolation)
        return resized


class Resize(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size
    The larger the original image is, the more times it takes to
    interpolate
    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
    """    
    """
    [CF] 固定尺寸缩放。
    
    将视频片段的所有帧缩放到指定的固定尺寸。
    
    Args:
        size (tuple): 目标尺寸 (width, height)
        interpolation (str): 插值方法
    """

    def __init__(self, size, interpolation='nearest'):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clip):
        resized = FF.resize_clip(
            clip, self.size, interpolation=self.interpolation)
        return resized


class RandomCrop(object):
    """Extract random crop at the same location for a list of images
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """    
    """
    [CF] 随机裁剪。
    
    在视频的每一帧上执行相同位置的随机裁剪。
    这是保持时空一致性的核心操作：所有帧共享同一个随机裁剪坐标。
    
    Args:
        size (int or tuple): 裁剪尺寸 (height, width)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(
                    im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = random.randint(0, im_w - w)
        y1 = random.randint(0, im_h - h)
        cropped = FF.crop_clip(clip, y1, x1, h, w)

        return cropped


class ThreeCrop(object):
    """Extract random crop at the same location for a list of images
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """
    """
    [CF] 三裁剪（测试阶段用）。
    
    对视频片段进行三个固定位置的裁剪（左/右 或 上/下），
    用于测试时的多视角评估，提升预测稳定性。
    
    Args:
        size (int or tuple): 裁剪尺寸
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        if w != im_w and h != im_h:
            clip = FF.resize_clip(clip, self.size, interpolation="bilinear")
            im_h, im_w, im_c = clip[0].shape

        step = np.max((np.max((im_w, im_h)) - self.size[0]) // 2, 0)
        cropped = []
        for i in range(3):
            if (im_h > self.size[0]):
                x1 = 0
                y1 = i * step
                cropped.extend(FF.crop_clip(clip, y1, x1, h, w))
            else:
                x1 = i * step
                y1 = 0
                cropped.extend(FF.crop_clip(clip, y1, x1, h, w))
        return cropped


class RandomRotation(object):
    """Rotate entire clip randomly by a random angle within
    given bounds
    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).
    """    
    """
    [CF] 随机旋转。
    
    在给定角度范围内随机选择一个角度，对视频的所有帧进行旋转。
    所有帧共享相同的旋转角度。
    
    Args:
        degrees (int or tuple): 角度范围。如果是单个数，范围为 (-degrees, degrees)
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')

        self.degrees = degrees

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        import skimage
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            rotated = [skimage.transform.rotate(img, angle) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        return rotated


class CenterCrop(object):
    """Extract center crop at the same location for a list of images
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """    
    """
    [CF] 中心裁剪（测试阶段用）。
    
    对视频片段的所有帧进行中心裁剪，提供确定性的输出。
    
    Args:
        size (int or tuple): 裁剪尺寸
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(
                    im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = int(round((im_w - w) / 2.))
        y1 = int(round((im_h - h) / 2.))
        cropped = FF.crop_clip(clip, y1, x1, h, w)

        return cropped


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation and hue of the clip
    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """
    """
    [CF] 颜色抖动（面向视频片段）。
    
    随机调整视频的亮度、对比度、饱和度和色调。
    所有帧共享相同的调整参数，操作顺序随机打乱以增加多样性。
    
    Args:
        brightness (float): 亮度抖动幅度
        contrast (float): 对比度抖动幅度
        saturation (float): 饱和度抖动幅度
        hue (float): 色调抖动幅度，范围 [0, 0.5]
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
        clip (list): list of PIL.Image
        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
        if isinstance(clip[0], np.ndarray):
            raise TypeError(
                'Color jitter not yet implemented for numpy arrays')
        elif isinstance(clip[0], PIL.Image.Image):
            brightness, contrast, saturation, hue = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)

            # Apply to all images
            jittered_clip = []
            for img in clip:
                for func in img_transforms:
                    jittered_img = func(img)
                jittered_clip.append(jittered_img)

        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        return jittered_clip


class Normalize(object):
    """Normalize a clip with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """    
    """
    [CF] 标准化。
    
    对张量格式的视频片段进行标准化：(clip - mean) / std。
    注意：此操作要求输入已经是 PyTorch 张量。
    
    Args:
        mean (sequence): 各通道均值
        std (sequence): 各通道标准差
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, clip):
        """
        Args:
            clip (Tensor): Tensor clip of size (T, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor clip.
        """
        return FF.normalize(clip, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

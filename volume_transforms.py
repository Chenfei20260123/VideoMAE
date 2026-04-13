# [CF] 2026-04-12:
# 这个文件定义了视频片段的张量转换操作。
# 核心功能：将多帧图像列表转换为标准的 (C, T, H, W) 格式张量。

import numpy as np
from PIL import Image
import torch


def convert_img(img):
    """
    [CF] 转换图像格式：从 (H, W, C) 转换为 (C, H, W)。
    
    这是为了适配 PyTorch 的通道优先（Channel First）格式。
    如果输入是灰度图（2维），则自动添加通道维度。
    """
    """Converts (H, W, C) numpy.ndarray to (C, W, H) format
    """
    if len(img.shape) == 3:
        img = img.transpose(2, 0, 1)
    if len(img.shape) == 2:
        img = np.expand_dims(img, 0)
    return img


class ClipToTensor(object):
    """Convert a list of m (H x W x C) numpy.ndarrays in the range [0, 255]
    to a torch.FloatTensor of shape (C x m x H x W) in the range [0, 1.0]
    """    
    """
    [CF] 将图像列表转换为 PyTorch 张量。
    
    输入：list of (H x W x C) numpy.ndarray 或 PIL.Image，像素范围 [0, 255]
    输出：torch.FloatTensor of shape (C x T x H x W)，像素范围 [0.0, 1.0]
    
    这是 VideoMAE 中最常用的视频张量化操作。
    """

    def __init__(self, channel_nb=3, div_255=True, numpy=False):
        self.channel_nb = channel_nb
        self.div_255 = div_255
        self.numpy = numpy

    def __call__(self, clip):
        """
        Args: clip (list of numpy.ndarray): clip (list of images)
        to be converted to tensor.
        """        
        """
        Args:
            clip: 图像列表，元素为 np.ndarray 或 PIL.Image
        Returns:
            形状为 (C, T, H, W) 的张量或 numpy 数组
        """
        # Retrieve shape
        # 获取图像尺寸
        if isinstance(clip[0], np.ndarray):
            h, w, ch = clip[0].shape
            assert ch == self.channel_nb, 'Got {0} instead of 3 channels'.format(
                ch)
        elif isinstance(clip[0], Image.Image):
            w, h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image\
            but got list of {0}'.format(type(clip[0])))
        
        # 预分配输出数组：形状为 (C, T, H, W)
        np_clip = np.zeros([self.channel_nb, len(clip), int(h), int(w)])

        # Convert
        # 逐帧转换并填充
        for img_idx, img in enumerate(clip):
            if isinstance(img, np.ndarray):
                pass
            elif isinstance(img, Image.Image):
                img = np.array(img, copy=False)
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image\
                but got list of {0}'.format(type(clip[0])))
            img = convert_img(img)
            np_clip[:, img_idx, :, :] = img
        if self.numpy:
            if self.div_255:
                np_clip = np_clip / 255.0
            return np_clip

        else:
            tensor_clip = torch.from_numpy(np_clip)

            if not isinstance(tensor_clip, torch.FloatTensor):
                tensor_clip = tensor_clip.float()
            if self.div_255:
                tensor_clip = torch.div(tensor_clip, 255)
            return tensor_clip


# Note this norms data to -1/1
class ClipToTensor_K(object):
    """Convert a list of m (H x W x C) numpy.ndarrays in the range [0, 255]
    to a torch.FloatTensor of shape (C x m x H x W) in the range [0, 1.0]
    """    
    """
    [CF] 将图像列表转换为 PyTorch 张量，并归一化到 [-1, 1] 范围。
    
    与 ClipToTensor 的区别：像素值会被映射到 [-1, 1] 而非 [0, 1]。
    公式：(pixel - 127.5) / 127.5
    
    某些预训练模型（如使用 Tanh 激活的生成模型）需要这种归一化方式。
    在 VideoMAE 的标准流程中，通常使用 ClipToTensor + GroupNormalize，
    因此这个类使用较少。
    """

    def __init__(self, channel_nb=3, div_255=True, numpy=False):
        self.channel_nb = channel_nb
        self.div_255 = div_255
        self.numpy = numpy

    def __call__(self, clip):
        """
        Args: clip (list of numpy.ndarray): clip (list of images)
        to be converted to tensor.
        """
        # 获取图像尺寸
        # Retrieve shape
        if isinstance(clip[0], np.ndarray):
            h, w, ch = clip[0].shape
            assert ch == self.channel_nb, 'Got {0} instead of 3 channels'.format(
                ch)
        elif isinstance(clip[0], Image.Image):
            w, h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image\
            but got list of {0}'.format(type(clip[0])))
        
        # 预分配输出数组
        np_clip = np.zeros([self.channel_nb, len(clip), int(h), int(w)])

        # Convert
        # 逐帧转换
        for img_idx, img in enumerate(clip):
            if isinstance(img, np.ndarray):
                pass
            elif isinstance(img, Image.Image):
                img = np.array(img, copy=False)
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image\
                but got list of {0}'.format(type(clip[0])))
            img = convert_img(img)
            np_clip[:, img_idx, :, :] = img
        if self.numpy:
            if self.div_255:
                np_clip = (np_clip - 127.5) / 127.5
            return np_clip

        else:
            tensor_clip = torch.from_numpy(np_clip)

            if not isinstance(tensor_clip, torch.FloatTensor):
                tensor_clip = tensor_clip.float()
            if self.div_255:
                tensor_clip = torch.div(torch.sub(tensor_clip, 127.5), 127.5)
            return tensor_clip


class ToTensor(object):
    """Converts numpy array to tensor
    """    
    """
    [CF] 简单的 numpy 到 tensor 转换器。
    
    不做任何归一化，只进行类型转换。
    """

    def __call__(self, array):
        tensor = torch.from_numpy(array)
        return tensor

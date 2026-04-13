# [CF] 2026-04-12:
# 这个文件定义了 VideoMAE 数据预处理的核心变换操作。
# 所有变换都是"Group"级别的，即对视频片段的每一帧应用完全相同的操作，
# 以确保视频的时空一致性。这是视频理解任务中数据增强的关键设计。

import torch
import torchvision.transforms.functional as F
import warnings
import random
import numpy as np
import torchvision
from PIL import Image, ImageOps
import numbers


class GroupRandomCrop(object):
    """
    [CF] 组随机裁剪：对视频片段的所有帧进行相同位置的随机裁剪。
    
    这是保持时空一致性的核心操作：所有帧共享同一个随机裁剪坐标，
    确保物体在帧间的相对位置关系保持不变。
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        
        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()
        # [CF] 关键：只随机生成一次裁剪坐标
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                # [CF] 所有帧使用相同的 (x1, y1) 进行裁剪
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return (out_images, label)


class GroupCenterCrop(object):
    """
    [CF] 组中心裁剪：对所有帧进行中心裁剪。
    通常用于验证/测试阶段，提供确定性的裁剪结果。
    """
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        return ([self.worker(img) for img in img_group], label)


class GroupNormalize(object):
    """
    [CF] 组标准化：对张量化的视频片段进行标准化。
    
    注意：输入是已经转换为 (C, T, H, W) 或 (T*C, H, W) 格式的张量。
    mean 和 std 会根据张量的实际通道数进行复制扩展。
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor_tuple):
        tensor, label = tensor_tuple
        rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
        rep_std = self.std * (tensor.size()[0]//len(self.std))
        
        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return (tensor,label)


class GroupGrayScale(object):
    """组灰度化：将所有帧转换为灰度图。"""
    def __init__(self, size):
        self.worker = torchvision.transforms.Grayscale(size)

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        return ([self.worker(img) for img in img_group], label)

    
class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    """
    [CF] 组缩放：将所有帧缩放到指定尺寸。
    
    保持长宽比的缩放：将短边缩放到 size，长边按比例缩放。
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        return ([self.worker(img) for img in img_group], label)


class GroupMultiScaleCrop(object):
    """
    [CF] 组多尺度裁剪：VideoMAE 预训练中最核心的空间数据增强。
    
    它结合了随机缩放和随机裁剪：
    1. 从预设的尺度列表中随机选择一个尺度
    2. 将图像缩放到该尺度
    3. 随机（或按固定偏移）裁剪出目标尺寸的区域
    
    这种增强让模型能够学习到对物体尺度不敏感的特征。
    """
    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        # scales: 尺度列表，相对于短边的比例，默认 [1, .875, .75, .66]
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort  # 允许的最大宽高失真程度
        self.fix_crop = fix_crop        # 是否使用固定的裁剪偏移（而非完全随机）
        self.more_fix_crop = more_fix_crop  # 是否使用更多固定位置（13个 vs 5个）
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        
        im_size = img_group[0].size
        # 采样裁剪尺寸和偏移
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        # 对所有帧应用相同的裁剪
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        # 缩放回目标尺寸
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation) for img in crop_img_group]
        return (ret_img_group, label)

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        # 根据短边计算各个尺度下的尺寸
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        # 如果计算出的尺寸与目标尺寸接近，则直接使用目标尺寸
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]
        # 生成所有有效的 (宽, 高) 组合
        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))
        # 随机选择一个尺寸组合
        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            # 完全随机的裁剪偏移
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            # 使用预设的固定偏移（5个或13个位置）
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        """
        [CF] 生成固定的裁剪偏移位置。
        将图像划分为 4x4 网格，从其中选择特定的交叉点作为裁剪起点。
        """
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        # 基础的 5 个位置：四角 + 中心
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            # 额外 8 个位置：四边中点 + 四个象限的中心
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter
        return ret


class Stack(object):
    """
    [CF] 堆叠操作：将图像列表沿通道维度堆叠成一个张量。
    
    对于 RGB 图像，输出形状为 (H, W, T*3)。
    这是 VideoMAE 将多帧图像合并为单个张量的关键步骤。
    """
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        
        if img_group[0].mode == 'L':
            return (np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2), label)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return (np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2), label)
            else:
                return (np.concatenate(img_group, axis=2), label)


class ToTorchFormatTensor(object):
    """
    [CF] 转换为 PyTorch 张量格式。
    
    将 PIL.Image (H x W x C) 或 numpy.ndarray 转换为
    torch.FloatTensor of shape (C x H x W)，并将像素值归一化到 [0.0, 1.0]。
    """
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic_tuple):
        pic, label = pic_tuple
        
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return (img.float().div(255.) if self.div else img.float(), label)


class IdentityTransform(object):

    def __call__(self, data):
        return data

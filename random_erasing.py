"""
This implementation is based on
https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/random_erasing.py
pulished under an Apache License 2.0.
"""

# [CF] 2026-04-12:
# 这个文件实现了 Random Erasing 数据增强策略。
# Random Erasing 通过在图像上随机擦除一个矩形区域（填充为噪声或常数），
# 来模拟物体遮挡，从而提升模型的鲁棒性和泛化能力。
# 在 VideoMAE 中，它主要用于微调阶段，作为一种额外的正则化手段。

import math
import random
import torch


def _get_pixels(
    per_pixel, rand_color, patch_size, dtype=torch.float32, device="cuda"
):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    """
    [CF] 生成用于填充擦除区域的像素值。
    
    支持三种填充模式：
    1. per_pixel=True: 每个像素独立地从正态分布采样（完全随机噪声）
    2. rand_color=True: 整个擦除块使用同一个随机颜色（每个通道一个值）
    3. 否则: 填充为常数 0（纯黑色）
    
    Args:
        per_pixel (bool): 是否为每个像素生成独立的随机值
        rand_color (bool): 是否为整个块生成一个随机颜色
        patch_size (tuple): 擦除区域的形状 (C, H, W)
        dtype: 数据类型
        device: 计算设备
    
    Returns:
        Tensor: 填充像素张量
    """
    # 注意：某些 CUDA 版本在 GPU 上执行 normal_() 可能导致非法内存访问
    # 如果遇到问题，可以将 normal_() 移到 CPU 上执行
    if per_pixel:
        # 每个像素独立采样：生成形状为 (C, H, W) 的正态分布随机数
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        # 每通道一个随机颜色：生成形状为 (C, 1, 1) 的正态分布随机数
        # 广播机制会自动将其扩展到整个 (H, W) 区域
        return torch.empty(
            (patch_size[0], 1, 1), dtype=dtype, device=device
        ).normal_()
    else:
        # 常数填充：全零（黑色）
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)


class RandomErasing:
    """Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """
    """
    [CF] 随机擦除数据增强。
    
    在图像上随机选择一个矩形区域，并用特定模式（常数/随机颜色/像素噪声）填充。
    可以生成多个擦除块，并特别支持视频数据的"立方体擦除"模式。
    
    论文: 'Random Erasing Data Augmentation' by Zhong et al.
    https://arxiv.org/pdf/1708.04896.pdf
    """
    def __init__(
        self,
        probability=0.5,          # 应用擦除的概率
        min_area=0.02,            # 擦除区域占图像面积的最小比例
        max_area=1 / 3,           # 擦除区域占图像面积的最大比例
        min_aspect=0.3,           # 擦除区域的最小宽高比
        max_aspect=None,          # 擦除区域的最大宽高比（默认为 1/min_aspect）
        mode="const",             # 填充模式: 'const', 'rand', 或 'pixel'
        min_count=1,              # 最少擦除块数量
        max_count=None,           # 最多擦除块数量
        num_splits=0,             # 批次中不应用擦除的前部样本数（用于保留干净样本）
        device="cuda",            # 计算设备
        cube=True,                # [VideoMAE 特有] 是否对批次中的所有帧使用相同的擦除位置
    ):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        # 宽高比在对数空间中均匀采样，保证对称性
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        self.cube = cube # [CF] 视频专用的"立方体擦除"模式
        if mode == "rand":
            self.rand_color = True  # per block random normal
        elif mode == "pixel":
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == "const"
        self.device = device

    def _erase(self, img, chan, img_h, img_w, dtype):
        """
        [CF] 对单张图像执行随机擦除（内部实现）。
        
        Args:
            img (Tensor): 单张图像，形状为 (C, H, W)
            chan, img_h, img_w: 通道数、高度、宽度
            dtype: 数据类型
        """
        # 根据概率决定是否跳过
        if random.random() > self.probability:
            return
        area = img_h * img_w
        # 确定擦除块的数量
        count = (
            self.min_count
            if self.min_count == self.max_count
            else random.randint(self.min_count, self.max_count)
        )
        for _ in range(count):
            # 最多尝试 10 次，找到合适的擦除区域
            for _ in range(10):
                # 计算目标擦除面积（总面积 × 比例 / 块数）
                target_area = (
                    random.uniform(self.min_area, self.max_area) * area / count
                )
                # 采样宽高比，并计算擦除区域的高和宽
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                # 确保擦除区域不超出图像边界
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    # 用生成的像素值填充擦除区域
                    img[:, top : top + h, left : left + w] = _get_pixels(
                        self.per_pixel,
                        self.rand_color,
                        (chan, h, w),
                        dtype=dtype,
                        device=self.device,
                    )
                    break

    def _erase_cube(
        self,
        img,
        batch_start,
        batch_size,
        chan,
        img_h,
        img_w,
        dtype,
    ):
        """
        [CF] 对批次中的所有（或部分）图像执行"立方体擦除"。
        
        这是 VideoMAE 对视频数据的特殊适配：
        对于同一批次中的所有帧（或从 batch_start 开始的帧），
        在完全相同的位置擦除相同大小的矩形区域。
        这模拟了视频中物体在连续帧中被持续遮挡的情况，保持了时空一致性。
        
        Args:
            img (Tensor): 批次图像，形状为 (B, C, H, W)
            batch_start (int): 开始应用擦除的样本索引
            batch_size (int): 批次大小
            chan, img_h, img_w: 通道数、高度、宽度
            dtype: 数据类型
        """
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = (
            self.min_count
            if self.min_count == self.max_count
            else random.randint(self.min_count, self.max_count)
        )
        for _ in range(count):
            for _ in range(100):
                target_area = (
                    random.uniform(self.min_area, self.max_area) * area / count
                )
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    for i in range(batch_start, batch_size):
                        img_instance = img[i]
                        img_instance[
                            :, top : top + h, left : left + w
                        ] = _get_pixels(
                            self.per_pixel,
                            self.rand_color,
                            (chan, h, w),
                            dtype=dtype,
                            device=self.device,
                        )
                    break

    def __call__(self, input):
        """
        [CF] 对输入应用随机擦除。
        
        支持两种输入格式：
        1. 单张图像：形状为 (C, H, W)
        2. 批次图像：形状为 (B, C, H, W)
        
        对于视频数据（批次中的多帧），可以通过 cube=True 启用"立方体擦除"，
        确保所有帧在相同位置被擦除，保持时空一致性。
        """
        if len(input.size()) == 3:
            self._erase(input, *input.size(), input.dtype)
        else:
            batch_size, chan, img_h, img_w = input.size()
            # skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = (
                batch_size // self.num_splits if self.num_splits > 1 else 0
            )
            if self.cube:
                self._erase_cube(
                    input,
                    batch_start,
                    batch_size,
                    chan,
                    img_h,
                    img_w,
                    input.dtype,
                )
            else:
                for i in range(batch_start, batch_size):
                    self._erase(input[i], chan, img_h, img_w, input.dtype)
        return input

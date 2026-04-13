"""
This implementation is based on
https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/auto_augment.py
pulished under an Apache License 2.0.

COMMENT FROM ORIGINAL:
AutoAugment, RandAugment, and AugMix for PyTorch
This code implements the searched ImageNet policies with various tweaks and
improvements and does not include any of the search code. AA and RA
Implementation adapted from:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py
AugMix adapted from:
    https://github.com/google-research/augmix
Papers:
    AutoAugment: Learning Augmentation Policies from Data
    https://arxiv.org/abs/1805.09501
    Learning Data Augmentation Strategies for Object Detection
    https://arxiv.org/abs/1906.11172
    RandAugment: Practical automated data augmentation...
    https://arxiv.org/abs/1909.13719
    AugMix: A Simple Data Processing Method to Improve Robustness and
    Uncertainty https://arxiv.org/abs/1912.02781

Hacked together by / Copyright 2020 Ross Wightman
"""

# [CF] 2026-04-12:
# 这个文件实现了 RandAugment 数据增强策略。
# RandAugment 是一种自动化的数据增强方法，它从一系列图像变换操作中随机选择，
# 并以统一的强度参数来应用它们，从而大幅提升模型的泛化能力。
# 在 VideoMAE 中，它主要用于微调阶段，对视频帧进行数据增强。

import math
import numpy as np
import random
import re
import PIL
from PIL import Image, ImageEnhance, ImageOps

# 获取 PIL 库的主版本号和次版本号，用于后续的版本兼容性处理
_PIL_VER = tuple([int(x) for x in PIL.__version__.split(".")[:2]])

# 默认的填充颜色（灰色），用于旋转、平移等操作后填补空白区域
_FILL = (128, 128, 128)

# 增强强度的最大级别。控制器 RNN 预测的值在 [0, _MAX_LEVEL] 范围内
# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.0

# 默认的超参数配置
_HPARAMS_DEFAULT = {
    "translate_const": 250,
    "img_mean": _FILL,
}

# 随机插值方法的选择范围（双线性 或 双三次）
_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


def _interpolation(kwargs):
    """
    [CF] 从参数中提取并处理插值方法。
    如果 'resample' 是一个列表或元组，则从中随机选择一个插值方法。
    这增加了数据增强的随机性。
    """
    interpolation = kwargs.pop("resample", Image.BILINEAR)
    if isinstance(interpolation, (list, tuple)):
        return random.choice(interpolation)
    else:
        return interpolation


def _check_args_tf(kwargs):
    """
    [CF] 检查并调整参数，以兼容 TensorFlow 的实现和不同 PIL 版本。
    主要处理两件事：
    1. 对于 PIL < 5.0，移除 'fillcolor' 参数（不被支持）
    2. 处理 'resample' 参数，允许随机选择插值方法
    """
    if "fillcolor" in kwargs and _PIL_VER < (5, 0):
        kwargs.pop("fillcolor")
    kwargs["resample"] = _interpolation(kwargs)

# =============================================================================
# [CF] 以下是一系列基础的图像变换操作函数
# 每个函数都接收一个 PIL.Image 和相应的变换参数，返回变换后的图像
# =============================================================================

def shear_x(img, factor, **kwargs):
    """
    [CF] X 轴方向的剪切变换。
    变换矩阵: [1, factor, 0, 0, 1, 0]
    factor 控制剪切的程度。
    """
    _check_args_tf(kwargs)
    return img.transform(
        img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs
    )


def shear_y(img, factor, **kwargs):
    """
    [CF] Y 轴方向的剪切变换。
    变换矩阵: [1, 0, 0, factor, 1, 0]
    """
    _check_args_tf(kwargs)
    return img.transform(
        img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs
    )


def translate_x_rel(img, pct, **kwargs):
    """
    [CF] X 轴方向的相对平移（相对于图像宽度的百分比）。
    pct: 平移比例，例如 0.1 表示平移图像宽度的 10%
    """
    pixels = pct * img.size[0]
    _check_args_tf(kwargs)
    return img.transform(
        img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs
    )


def translate_y_rel(img, pct, **kwargs):
    """
    [CF] Y 轴方向的相对平移（相对于图像高度的百分比）。
    """
    pixels = pct * img.size[1]
    _check_args_tf(kwargs)
    return img.transform(
        img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs
    )


def translate_x_abs(img, pixels, **kwargs):
    """
    [CF] X 轴方向的绝对平移（以像素为单位）。
    """
    _check_args_tf(kwargs)
    return img.transform(
        img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs
    )


def translate_y_abs(img, pixels, **kwargs):
    """
    [CF] Y 轴方向的绝对平移（以像素为单位）。
    """
    _check_args_tf(kwargs)
    return img.transform(
        img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs
    )


def rotate(img, degrees, **kwargs):
    """
    [CF] 旋转图像。
    对于不同 PIL 版本做了兼容处理：
    - PIL >= 5.2: 直接使用 img.rotate
    - PIL >= 5.0: 手动构建仿射变换矩阵实现旋转
    - 旧版本: 使用 img.rotate 并指定 resample
    """
    _check_args_tf(kwargs)
    if _PIL_VER >= (5, 2):
        return img.rotate(degrees, **kwargs)
    elif _PIL_VER >= (5, 0):
        w, h = img.size
        post_trans = (0, 0)
        rotn_center = (w / 2.0, h / 2.0)
        angle = -math.radians(degrees)
        matrix = [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,
        ]

        def transform(x, y, matrix):
            (a, b, c, d, e, f) = matrix
            return a * x + b * y + c, d * x + e * y + f

        matrix[2], matrix[5] = transform(
            -rotn_center[0] - post_trans[0],
            -rotn_center[1] - post_trans[1],
            matrix,
        )
        matrix[2] += rotn_center[0]
        matrix[5] += rotn_center[1]
        return img.transform(img.size, Image.AFFINE, matrix, **kwargs)
    else:
        return img.rotate(degrees, resample=kwargs["resample"])


def auto_contrast(img, **__):
    """
    [CF] 自动对比度：归一化图像直方图，增强对比度。
    """
    return ImageOps.autocontrast(img)


def invert(img, **__):
    """
    [CF] 反色：将每个像素的颜色取反。
    """
    return ImageOps.invert(img)


def equalize(img, **__):
    """
    [CF] 直方图均衡化：使图像的亮度分布更均匀。
    """
    return ImageOps.equalize(img)


def solarize(img, thresh, **__):
    """
    [CF] 曝光过度：将高于阈值 thresh 的像素取反。
    """
    return ImageOps.solarize(img, thresh)


def solarize_add(img, add, thresh=128, **__):
    """
    [CF] 加法曝光：将低于阈值 thresh 的像素值增加 add。
    通过构建查找表（LUT）实现。
    """
    lut = []
    for i in range(256):
        if i < thresh:
            lut.append(min(255, i + add))
        else:
            lut.append(i)
    if img.mode in ("L", "RGB"):
        if img.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return img.point(lut)
    else:
        return img


def posterize(img, bits_to_keep, **__):
    """
    [CF] 色调分离：减少每个通道的比特数，产生颜色分层效果。
    bits_to_keep: 保留的比特数（1-8）
    """
    if bits_to_keep >= 8:
        return img
    return ImageOps.posterize(img, bits_to_keep)


def contrast(img, factor, **__):
    """
    [CF] 调整对比度。factor=1.0 表示不变。
    """
    return ImageEnhance.Contrast(img).enhance(factor)


def color(img, factor, **__):
    """
    [CF] 调整色彩饱和度。factor=1.0 表示不变。
    """
    return ImageEnhance.Color(img).enhance(factor)


def brightness(img, factor, **__):
    """
    [CF] 调整亮度。factor=1.0 表示不变。
    """
    return ImageEnhance.Brightness(img).enhance(factor)


def sharpness(img, factor, **__):
    """
    [CF] 调整锐度。factor=1.0 表示不变。
    """
    return ImageEnhance.Sharpness(img).enhance(factor)


def _randomly_negate(v):
    """With 50% prob, negate the value"""
    """
    [CF] 以 50% 的概率对值取反。
    用于让增强操作可以双向进行（如顺时针/逆时针旋转）。
    """
    return -v if random.random() > 0.5 else v

# =============================================================================
# [CF] 以下是"级别到参数"的转换函数
# RandAugment 的核心思想是：所有操作共享一个统一的强度级别 (level)，
# 但不同操作需要将这个级别映射到不同的实际参数范围。
# 例如：level=5 对于旋转可能对应 15 度，对于亮度可能对应 1.5 倍。
# =============================================================================

def _rotate_level_to_arg(level, _hparams):
    # range [-30, 30]
    """旋转: level 映射到 [-30, 30] 度"""
    level = (level / _MAX_LEVEL) * 30.0
    level = _randomly_negate(level)
    return (level,)


def _enhance_level_to_arg(level, _hparams):
    """增强类操作（对比度、亮度等）: level 映射到 [0.1, 1.9]"""
    # range [0.1, 1.9]
    return ((level / _MAX_LEVEL) * 1.8 + 0.1,)


def _enhance_increasing_level_to_arg(level, _hparams):
    """
    [CF] 增强类操作（递增版本）: level 映射到以 1.0 为中心的对称范围。
    level 增大时，变换强度向两个方向（减弱或增强）增加。
    """
    # the 'no change' level is 1.0, moving away from that towards 0. or 2.0 increases the enhancement blend
    # range [0.1, 1.9]
    level = (level / _MAX_LEVEL) * 0.9
    level = 1.0 + _randomly_negate(level)
    return (level,)


def _shear_level_to_arg(level, _hparams):
    """剪切: level 映射到 [-0.3, 0.3]"""
    # range [-0.3, 0.3]
    level = (level / _MAX_LEVEL) * 0.3
    level = _randomly_negate(level)
    return (level,)


def _translate_abs_level_to_arg(level, hparams):
    """绝对平移: level 映射到 [-translate_const, translate_const] 像素"""
    translate_const = hparams["translate_const"]
    level = (level / _MAX_LEVEL) * float(translate_const)
    level = _randomly_negate(level)
    return (level,)


def _translate_rel_level_to_arg(level, hparams):
    """相对平移: level 映射到 [-translate_pct, translate_pct]（图像尺寸的百分比）"""
    # default range [-0.45, 0.45]
    translate_pct = hparams.get("translate_pct", 0.45)
    level = (level / _MAX_LEVEL) * translate_pct
    level = _randomly_negate(level)
    return (level,)


def _posterize_level_to_arg(level, _hparams):
    """色调分离: level 越大，保留的比特数越少，效果越强"""
    # As per Tensorflow TPU EfficientNet impl
    # range [0, 4], 'keep 0 up to 4 MSB of original image'
    # intensity/severity of augmentation decreases with level
    return (int((level / _MAX_LEVEL) * 4),)


def _posterize_increasing_level_to_arg(level, hparams):
    """色调分离（递增版本）"""
    # As per Tensorflow models research and UDA impl
    # range [4, 0], 'keep 4 down to 0 MSB of original image',
    # intensity/severity of augmentation increases with level
    return (4 - _posterize_level_to_arg(level, hparams)[0],)


def _posterize_original_level_to_arg(level, _hparams):
    """色调分离（原始版本）: 保留 4-8 比特"""
    # As per original AutoAugment paper description
    # range [4, 8], 'keep 4 up to 8 MSB of image'
    # intensity/severity of augmentation decreases with level
    return (int((level / _MAX_LEVEL) * 4) + 4,)


def _solarize_level_to_arg(level, _hparams):
    """曝光过度: level 映射到 [0, 256] 的阈值"""
    # range [0, 256]
    # intensity/severity of augmentation decreases with level
    return (int((level / _MAX_LEVEL) * 256),)


def _solarize_increasing_level_to_arg(level, _hparams):
    """曝光过度（递增版本）"""
    # range [0, 256]
    # intensity/severity of augmentation increases with level
    return (256 - _solarize_level_to_arg(level, _hparams)[0],)


def _solarize_add_level_to_arg(level, _hparams):
    """加法曝光: level 映射到 [0, 110] 的加数"""
    # range [0, 110]
    return (int((level / _MAX_LEVEL) * 110),)

# [CF] 映射表：操作名称 -> 级别转换函数
# RandAugment 通过统一的级别参数，调用这些函数将其转换为具体操作的参数
LEVEL_TO_ARG = {
    "AutoContrast": None,
    "Equalize": None,
    "Invert": None,
    "Rotate": _rotate_level_to_arg,
    # There are several variations of the posterize level scaling in various Tensorflow/Google repositories/papers
    "Posterize": _posterize_level_to_arg,
    "PosterizeIncreasing": _posterize_increasing_level_to_arg,
    "PosterizeOriginal": _posterize_original_level_to_arg,
    "Solarize": _solarize_level_to_arg,
    "SolarizeIncreasing": _solarize_increasing_level_to_arg,
    "SolarizeAdd": _solarize_add_level_to_arg,
    "Color": _enhance_level_to_arg,
    "ColorIncreasing": _enhance_increasing_level_to_arg,
    "Contrast": _enhance_level_to_arg,
    "ContrastIncreasing": _enhance_increasing_level_to_arg,
    "Brightness": _enhance_level_to_arg,
    "BrightnessIncreasing": _enhance_increasing_level_to_arg,
    "Sharpness": _enhance_level_to_arg,
    "SharpnessIncreasing": _enhance_increasing_level_to_arg,
    "ShearX": _shear_level_to_arg,
    "ShearY": _shear_level_to_arg,
    "TranslateX": _translate_abs_level_to_arg,
    "TranslateY": _translate_abs_level_to_arg,
    "TranslateXRel": _translate_rel_level_to_arg,
    "TranslateYRel": _translate_rel_level_to_arg,
}

# [CF] 映射表：操作名称 -> 实际的图像处理函数
NAME_TO_OP = {
    "AutoContrast": auto_contrast,
    "Equalize": equalize,
    "Invert": invert,
    "Rotate": rotate,
    "Posterize": posterize,
    "PosterizeIncreasing": posterize,
    "PosterizeOriginal": posterize,
    "Solarize": solarize,
    "SolarizeIncreasing": solarize,
    "SolarizeAdd": solarize_add,
    "Color": color,
    "ColorIncreasing": color,
    "Contrast": contrast,
    "ContrastIncreasing": contrast,
    "Brightness": brightness,
    "BrightnessIncreasing": brightness,
    "Sharpness": sharpness,
    "SharpnessIncreasing": sharpness,
    "ShearX": shear_x,
    "ShearY": shear_y,
    "TranslateX": translate_x_abs,
    "TranslateY": translate_y_abs,
    "TranslateXRel": translate_x_rel,
    "TranslateYRel": translate_y_rel,
}

# [CF] ============================================================================
# [CF] 第二部分：RandAugment 的组装与执行逻辑
# [CF] 核心组件：AugmentOp（单个增强操作）、RandAugment（整体增强流水线）
# [CF] ============================================================================
class AugmentOp:
    """
    Apply for video.
    """
    """
    [CF] 单个增强操作的封装类。
    
    它将第一部分定义的"操作函数"和"级别转换函数"绑定在一起，
    形成一个可调用的对象。每次调用时，它会：
    1. 根据概率决定是否执行该操作
    2. 将统一的强度 magnitude 转换为该操作的具体参数
    3. 对输入的图像（或图像列表）执行变换
    """
    def __init__(self, name, prob=0.5, magnitude=10, hparams=None):
        """
        Args:
            name (str): 操作名称，如 'Rotate', 'Color'，必须在 NAME_TO_OP 中
            prob (float): 该操作被应用的概率（0-1）
            magnitude (float): 统一的增强强度（0-10）
            hparams (dict): 超参数，包含填充色、插值方法等
        """
        hparams = hparams or _HPARAMS_DEFAULT
        # 从映射表中获取实际的图像处理函数
        self.aug_fn = NAME_TO_OP[name]
        # 从映射表中获取级别转换函数
        self.level_fn = LEVEL_TO_ARG[name]
        self.prob = prob
        self.magnitude = magnitude
        self.hparams = hparams.copy()
        # 构建传递给 aug_fn 的关键字参数
        self.kwargs = {
            "fillcolor": hparams["img_mean"]
            if "img_mean" in hparams
            else _FILL,
            "resample": hparams["interpolation"]
            if "interpolation" in hparams
            else _RANDOM_INTERPOLATION,
        }

        # If magnitude_std is > 0, we introduce some randomness
        # in the usually fixed policy and sample magnitude from a normal distribution
        # with mean `magnitude` and std-dev of `magnitude_std`.
        # NOTE This is my own hack, being tested, not in papers or reference impls.
        # [CF] 如果 magnitude_std > 0，则从正态分布中采样强度
        # 这为固定的策略引入了一些随机性，进一步增强数据多样性
        # 注意：这是作者自己的实验性功能，并非原论文的一部分
        self.magnitude_std = self.hparams.get("magnitude_std", 0)

    def __call__(self, img_list):
        """
        [CF] 对输入的图像或图像列表应用增强操作。
        
        特别适配了 VideoMAE 的场景：输入可以是一个 PIL Image 列表（视频帧），
        此时会对列表中的每一帧应用完全相同的变换，以保持时空一致性。
        """
        # 步骤1: 根据概率决定是否跳过该操作
        if self.prob < 1.0 and random.random() > self.prob:
            return img_list
        
        # 步骤2: 确定本次应用的强度
        magnitude = self.magnitude
        if self.magnitude_std and self.magnitude_std > 0:
            # 从正态分布中采样，增加随机性
            magnitude = random.gauss(magnitude, self.magnitude_std)
        magnitude = min(_MAX_LEVEL, max(0, magnitude))  # clip to valid range
        # 步骤3: 将统一强度转换为该操作的具体参数
        level_args = (
            self.level_fn(magnitude, self.hparams)
            if self.level_fn is not None
            else () # 某些操作（如 AutoContrast）不需要参数
        )
        # 步骤4: 应用变换
        if isinstance(img_list, list):
            # 对于视频（图像列表），对每一帧应用完全相同的变换
            return [
                self.aug_fn(img, *level_args, **self.kwargs) for img in img_list
            ]
        else:
            # 对于单张图像
            return self.aug_fn(img_list, *level_args, **self.kwargs)

# [CF] 默认的 RandAugment 操作池（共 15 种操作）
# 这些操作被原论文验证为对图像分类任务有效
_RAND_TRANSFORMS = [
    "AutoContrast",
    "Equalize",
    "Invert",
    "Rotate",
    "Posterize",
    "Solarize",
    "SolarizeAdd",
    "Color",
    "Contrast",
    "Brightness",
    "Sharpness",
    "ShearX",
    "ShearY",
    "TranslateXRel",
    "TranslateYRel",
]

# [CF] "递增"版本的操作池
# 这些操作的强度随着 magnitude 的增加而单调递增（更符合直觉）
_RAND_INCREASING_TRANSFORMS = [
    "AutoContrast",
    "Equalize",
    "Invert",
    "Rotate",
    "PosterizeIncreasing",
    "SolarizeIncreasing",
    "SolarizeAdd",
    "ColorIncreasing",
    "ContrastIncreasing",
    "BrightnessIncreasing",
    "SharpnessIncreasing",
    "ShearX",
    "ShearY",
    "TranslateXRel",
    "TranslateYRel",
]

# [CF] 操作选择的权重（实验性）
# 根据论文中提到的各操作的相对提升效果粗略设定
# 使用权重可以让更有效的操作被更频繁地选中
# These experimental weights are based loosely on the relative improvements mentioned in paper.
# They may not result in increased performance, but could likely be tuned to so.
_RAND_CHOICE_WEIGHTS_0 = {
    "Rotate": 0.3,
    "ShearX": 0.2,
    "ShearY": 0.2,
    "TranslateXRel": 0.1,
    "TranslateYRel": 0.1,
    "Color": 0.025,
    "Sharpness": 0.025,
    "AutoContrast": 0.025,
    "Solarize": 0.005,
    "SolarizeAdd": 0.005,
    "Contrast": 0.005,
    "Brightness": 0.005,
    "Equalize": 0.005,
    "Posterize": 0,
    "Invert": 0,
}


def _select_rand_weights(weight_idx=0, transforms=None):
    """
    [CF] 根据权重索引生成操作选择的概率分布。
    目前只实现了 weight_idx=0 这一组权重。
    """
    transforms = transforms or _RAND_TRANSFORMS
    assert weight_idx == 0  # only one set of weights currently
    rand_weights = _RAND_CHOICE_WEIGHTS_0
    probs = [rand_weights[k] for k in transforms]
    probs /= np.sum(probs)
    return probs


def rand_augment_ops(magnitude=10, hparams=None, transforms=None):
    """
    [CF] 创建 RandAugment 的操作池。
    
    为 transforms 列表中的每个操作名称创建一个 AugmentOp 实例。
    所有操作共享相同的 magnitude，但各自有独立的应用概率。
    """
    hparams = hparams or _HPARAMS_DEFAULT
    transforms = transforms or _RAND_TRANSFORMS
    return [
        AugmentOp(name, prob=0.5, magnitude=magnitude, hparams=hparams)
        for name in transforms
    ]


class RandAugment:
    """
    [CF] RandAugment 的主类。
    
    它管理一个操作池，并在每次被调用时：
    1. 从操作池中随机选择 num_layers 个操作（可放回/不放回）
    2. 将选中的操作依次应用到输入图像上
    """
    def __init__(self, ops, num_layers=2, choice_weights=None):
        """
        Args:
            ops (list): AugmentOp 实例的列表（操作池）
            num_layers (int): 每张图像应用的增强操作数量
            choice_weights (np.ndarray): 选择各操作的概率权重
        """
        self.ops = ops
        self.num_layers = num_layers
        self.choice_weights = choice_weights

    def __call__(self, img):
        # no replacement when using weighted choice
        """
        [CF] 对输入图像应用 RandAugment 增强。
        """
        # 从操作池中随机选择 num_layers 个操作
        # 如果提供了 choice_weights，则按权重采样且不放回
        ops = np.random.choice(
            self.ops,
            self.num_layers,
            replace=self.choice_weights is None,
            p=self.choice_weights,
        )
        # 依次应用选中的操作
        for op in ops:
            img = op(img)
        return img


def rand_augment_transform(config_str, hparams):
    """
    RandAugment: Practical automated data augmentation... - https://arxiv.org/abs/1909.13719

    Create a RandAugment transform
    :param config_str: String defining configuration of random augmentation. Consists of multiple sections separated by
    dashes ('-'). The first section defines the specific variant of rand augment (currently only 'rand'). The remaining
    sections, not order sepecific determine
        'm' - integer magnitude of rand augment
        'n' - integer num layers (number of transform ops selected per image)
        'w' - integer probabiliy weight index (index of a set of weights to influence choice of op)
        'mstd' -  float std deviation of magnitude noise applied
        'inc' - integer (bool), use augmentations that increase in severity with magnitude (default: 0)
    Ex 'rand-m9-n3-mstd0.5' results in RandAugment with magnitude 9, num_layers 3, magnitude_std 0.5
    'rand-mstd1-w0' results in magnitude_std 1.0, weights 0, default magnitude of 10 and num_layers 2
    :param hparams: Other hparams (kwargs) for the RandAugmentation scheme
    :return: A PyTorch compatible Transform
    """
    """
    [CF] RandAugment 的工厂函数。
    
    解析配置字符串，创建并返回一个配置好的 RandAugment 实例。
    
    配置字符串格式示例：
        'rand-m9-n3-mstd0.5' -> magnitude=9, num_layers=3, magnitude_std=0.5
        'rand-mstd1-w0' -> magnitude_std=1.0, weight_idx=0, 其他用默认值
    
    Args:
        config_str (str): 配置字符串，以 'rand' 开头，后续用 '-' 分隔参数
        hparams (dict): 其他超参数
    
    Returns:
        RandAugment: 配置好的增强对象
    """
    # 默认配置
    magnitude = _MAX_LEVEL  # default to _MAX_LEVEL for magnitude (currently 10)
    num_layers = 2  # default to 2 ops per image
    weight_idx = None  # default to no probability weights for op choice
    transforms = _RAND_TRANSFORMS
    # 解析配置字符串
    config = config_str.split("-")
    assert config[0] == "rand"
    config = config[1:]
    for c in config:
        cs = re.split(r"(\d.*)", c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == "mstd":
            # 强度噪声的标准差
            # noise param injected via hparams for now
            hparams.setdefault("magnitude_std", float(val))
        elif key == "inc":
            if bool(val):
                transforms = _RAND_INCREASING_TRANSFORMS
        elif key == "m":
            magnitude = int(val)
        elif key == "n":
            num_layers = int(val)
        elif key == "w":
            weight_idx = int(val)
        else:
            assert NotImplementedError
    ra_ops = rand_augment_ops(
        magnitude=magnitude, hparams=hparams, transforms=transforms
    )
    choice_weights = (
        None if weight_idx is None else _select_rand_weights(weight_idx)
    )
    return RandAugment(ra_ops, num_layers, choice_weights=choice_weights)

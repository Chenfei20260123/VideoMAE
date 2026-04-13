# [CF] 2026-04-10:
# 这个文件实现了 Mixup 和 CutMix 两种数据增强策略。
# 它们都是通过"混合"不同的训练样本来创建新的、更具挑战性的训练数据，
# 从而作为一种强大的正则化手段，提升模型的泛化能力。
# 在 VideoMAE 中，这个文件主要用于微调（finetuning）阶段。

""" Mixup and Cutmix

Papers:
mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)

CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features (https://arxiv.org/abs/1905.04899)

Code Reference:
CutMix: https://github.com/clovaai/CutMix-PyTorch

Hacked together by / Copyright 2019, Ross Wightman
"""
import numpy as np
import torch


def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    """
    [CF] 将标签转换为 one-hot 编码格式。
    
    Args:
        x (Tensor): 原始标签张量，形状为 [N] 或 [N, 1]。
        num_classes (int): 总类别数。
        on_value (float): 目标类别的值（通常为 1.0 或平滑后的值）。
        off_value (float): 非目标类别的值（通常为 0.0 或平滑值）。
        device (str): 计算设备。
    
    Returns:
        Tensor: One-hot 编码后的标签，形状为 [N, num_classes]。
    """
    x = x.long().view(-1, 1)
    # [CF] 创建一个全为 off_value 的张量，然后在目标位置填充 on_value
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)


def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cuda'):
    """
    [CF] 根据 mixup/cutmix 的混合比例 lam，计算混合后的标签。
    
    核心思想：对于两个样本的混合，其标签也按照相同的比例 lam 进行线性混合。
    
    Args:
        target (Tensor): 原始标签，形状为 [N]。
        num_classes (int): 总类别数。
        lam (float or Tensor): 混合比例。lam 表示第一个样本的权重，(1-lam) 表示第二个样本的权重。
        smoothing (float): 标签平滑系数。
        device (str): 计算设备。
    
    Returns:
        Tensor: 混合后的软标签，形状为 [N, num_classes]。
    """
    # [CF] 计算标签平滑后的 on_value 和 off_value
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    # [CF] 将原始标签转换为 one-hot 编码（已经包含标签平滑）
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    # [CF] 将批次"翻转"作为第二个样本的标签。即第 i 个样本与第 N-i-1 个样本混合。
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)
    # [CF] 按照比例 lam 混合两个标签
    return y1 * lam + y2 * (1. - lam)


def rand_bbox(img_shape, lam, margin=0., count=None):
    """ Standard CutMix bounding-box
    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.

    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    
    rand_bbox 函数是 CutMix 数据增强的核心执行者。它的应用目的非常明确：根据给定的混合比例 lam，在图像上随机生成一个矩形区域（边界框），用于后续的"剪切-粘贴"操作。
    核心应用目的：
    在 CutMix 中，我们不是像 Mixup 那样对整个图像进行像素级混合，而是：
    1. 从图像 A 中切掉一块矩形区域
    2. 将图像 B 中对应位置的矩形区域粘贴过来
    rand_bbox 函数就是负责第 1 步和第 2 步中"确定矩形区域位置和大小" 的关键函数。

    [CF] 标准 CutMix 边界框生成器。
    
    根据混合比例 lam 生成一个随机的正方形边界框。边界框的面积比例约等于 1 - lam。
    
    Args:
        img_shape (tuple): 图像形状，通常为 (..., H, W)。
        lam (float): CutMix 的 lambda 值（目标混合比例）。
        margin (float): 边界框边距的百分比，用于防止边界框超出图像。
        count (int): 生成的边界框数量。
    
    Returns:
        tuple: (yl, yh, xl, xh) 边界框的 y 轴和 x 轴的起止坐标。
    """
    # [CF] 根据 lam 计算边界框的边长比例。面积 = ratio^2 ≈ 1 - lam
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    # [CF] 应用边距限制
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    # [CF] 随机选择边界框的中心点
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    # [CF] 根据中心点和尺寸计算边界框的起止坐标，并裁剪到图像范围内
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh


def rand_bbox_minmax(img_shape, minmax, count=None):
    """ Min-Max CutMix bounding-box
    Inspired by Darknet cutmix impl, generates a random rectangular bbox
    based on min/max percent values applied to each dimension of the input image.

    Typical defaults for minmax are usually in the  .2-.3 for min and .8-.9 range for max.

    Args:
        img_shape (tuple): Image shape as tuple
        minmax (tuple or list): Min and max bbox ratios (as percent of image size)
        count (int): Number of bbox to generate

    [CF] Min-Max CutMix 边界框生成器。
    
    与标准 CutMix 不同，这种方法直接指定边界框尺寸相对于图像尺寸的最小和最大比例。
    
    Args:
        img_shape (tuple): 图像形状。
        minmax (tuple or list): (min_ratio, max_ratio)，边界框边长的最小和最大比例。
        count (int): 生成的边界框数量。
    
    Returns:
        tuple: (yl, yu, xl, xu) 边界框坐标。
    """
    assert len(minmax) == 2
    img_h, img_w = img_shape[-2:]
    # [CF] 在 [min_ratio, max_ratio] 范围内随机选择边界框的高度和宽度
    cut_h = np.random.randint(int(img_h * minmax[0]), int(img_h * minmax[1]), size=count)
    cut_w = np.random.randint(int(img_w * minmax[0]), int(img_w * minmax[1]), size=count)
    # [CF] 随机选择边界框的左上角坐标
    yl = np.random.randint(0, img_h - cut_h, size=count)
    xl = np.random.randint(0, img_w - cut_w, size=count)
    yu = yl + cut_h
    xu = xl + cut_w
    return yl, yu, xl, xu


def cutmix_bbox_and_lam(img_shape, lam, ratio_minmax=None, correct_lam=True, count=None):
    """ Generate bbox and apply lambda correction.

    [CF] 生成 CutMix 边界框，并可选地对 lambda 进行修正。
    
    如果边界框因为靠近图像边缘而被裁剪，其实际面积会小于预期，
    此时 correct_lam=True 会根据实际面积重新计算 lambda。
    
    Args:
        img_shape (tuple): 图像形状。
        lam (float): 期望的混合比例。
        ratio_minmax (tuple, optional): 如果提供，则使用 min-max 方式生成边界框。
        correct_lam (bool): 是否根据实际裁剪区域修正 lambda。
        count (int): 边界框数量。
    
    Returns:
        tuple: (边界框坐标元组, 修正后的 lam)
    """
    if ratio_minmax is not None:
        yl, yu, xl, xu = rand_bbox_minmax(img_shape, ratio_minmax, count=count)
    else:
        yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    if correct_lam or ratio_minmax is not None:
        # [CF] 计算实际边界框面积，并据此修正 lambda
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1. - bbox_area / float(img_shape[-2] * img_shape[-1])
    return (yl, yu, xl, xu), lam


class Mixup:
    """ Mixup/Cutmix that applies different params to each element or whole batch

    Args:
        mixup_alpha (float): mixup alpha value, mixup is active if > 0.
        cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
        cutmix_minmax (List[float]): cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None.
        prob (float): probability of applying mixup or cutmix per batch or element
        switch_prob (float): probability of switching to cutmix instead of mixup when both are active
        mode (str): how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
        correct_lam (bool): apply lambda correction when cutmix bbox clipped by image borders
        label_smoothing (float): apply label smoothing to the mixed target tensor
        num_classes (int): number of classes for target

    [CF] Mixup/Cutmix 主类，支持以不同粒度（批次级、元素对级、元素级）应用数据混合。
    
    在 VideoMAE 微调阶段，通常使用批次级 (mode='batch') 的 Mixup。
    """
    def __init__(self, mixup_alpha=1., cutmix_alpha=0., cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                 mode='batch', correct_lam=True, label_smoothing=0.1, num_classes=1000):
        """
        [CF] 初始化 Mixup 类。
        
        Args:
            mixup_alpha (float): Mixup 的 Beta 分布参数，>0 时启用 Mixup。
            cutmix_alpha (float): Cutmix 的 Beta 分布参数，>0 时启用 Cutmix。
            cutmix_minmax (List[float]): 如果提供，则使用 min-max 模式而非 alpha。
            prob (float): 对每个批次/元素应用混合的概率。
            switch_prob (float): 当两者都启用时，切换到 Cutmix 的概率。
            mode (str): 混合模式，'batch', 'pair', 或 'elem'。
            correct_lam (bool): 是否修正 CutMix 中因裁剪而变化的 lambda。
            label_smoothing (float): 标签平滑系数。
            num_classes (int): 类别总数。
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            # [CF] 当使用 minmax 模式时，强制 cutmix_alpha = 1.0 以简化逻辑
            # force cutmix alpha == 1.0 when minmax active to keep logic simple & safe
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode
        self.correct_lam = correct_lam  # correct lambda based on clipped area for cutmix
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop) # [CF] 可由训练循环控制是否启用混合

    def _params_per_elem(self, batch_size):
        """
        [CF] 为批次中的每个元素独立生成混合参数。
        
        Returns:
            lam (np.ndarray): 形状为 [batch_size] 的 lambda 数组。
            use_cutmix (np.ndarray): 形状为 [batch_size] 的布尔数组，指示是否使用 CutMix。
        """
        lam = np.ones(batch_size, dtype=np.float32)
        use_cutmix = np.zeros(batch_size, dtype=np.bool)
        if self.mixup_enabled:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                # [CF] 两者都启用：以 switch_prob 的概率选择 Cutmix
                use_cutmix = np.random.rand(batch_size) < self.switch_prob
                lam_mix = np.where(
                    use_cutmix,
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size),
                    np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size))
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)
            elif self.cutmix_alpha > 0.:
                use_cutmix = np.ones(batch_size, dtype=np.bool)
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            # [CF] 以 mix_prob 的概率决定是否真正应用混合
            lam = np.where(np.random.rand(batch_size) < self.mix_prob, lam_mix.astype(np.float32), lam)
        return lam, use_cutmix

    def _params_per_batch(self):
        """
        [CF] 为整个批次生成一个统一的混合参数。
        
        Returns:
            lam (float): 混合比例 lambda。
            use_cutmix (bool): 是否使用 CutMix。
        """
        lam = 1.
        use_cutmix = False
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if use_cutmix else \
                    np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
        return lam, use_cutmix

    def _mix_elem(self, x):
        
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size)
        x_orig = x.clone()  # need to keep an unmodified original for mixing source
        for i in range(batch_size):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        x[i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    x[i][..., yl:yh, xl:xh] = x_orig[j][..., yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)
        return torch.tensor(lam_batch, device=x.device, dtype=x.dtype).unsqueeze(1)

    def _mix_pair(self, x):
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size // 2)
        x_orig = x.clone()  # need to keep an unmodified original for mixing source
        for i in range(batch_size // 2):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        x[i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    x[i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                    x[j][:, yl:yh, xl:xh] = x_orig[i][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)
                    x[j] = x[j] * lam + x_orig[i] * (1 - lam)
        lam_batch = np.concatenate((lam_batch, lam_batch[::-1]))
        return torch.tensor(lam_batch, device=x.device, dtype=x.dtype).unsqueeze(1)

    def _mix_batch(self, x):
        """
        [CF] 批次级混合：对整个批次应用相同的混合策略。
        
        这是最常用的模式。将批次 x 与其翻转后的版本 x.flip(0) 进行混合。
        """
        lam, use_cutmix = self._params_per_batch()
        if lam == 1.:
            return 1.
        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                x.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
            x[..., yl:yh, xl:xh] = x.flip(0)[..., yl:yh, xl:xh]
        else:
            x_flipped = x.flip(0).mul_(1. - lam)
            x.mul_(lam).add_(x_flipped)
        return lam

    def __call__(self, x, target):
        """
        [CF] 对输入数据和标签应用 Mixup/Cutmix。
        
        Args:
            x (Tensor): 输入数据，形状为 [B, ...]。
            target (Tensor): 标签，形状为 [B]。
        
        Returns:
            tuple: (mixed_x, mixed_target)
        """
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        if self.mode == 'elem':
            lam = self._mix_elem(x)
        elif self.mode == 'pair':
            lam = self._mix_pair(x)
        else:
            lam = self._mix_batch(x)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing, x.device)
        return x, target

# [CF] ============================================================================
# [CF] 以下 FastCollateMixup 类是为了与特定的数据加载方式兼容而设计的，
# [CF] 在 VideoMAE 的标准实现中通常不会用到，因此这里省略了详细注释。
# [CF] 它的核心逻辑与上面的 Mixup 类相似，只是在数据收集（collate）阶段执行混合。
# [CF] ============================================================================
class FastCollateMixup(Mixup):
    """ Fast Collate w/ Mixup/Cutmix that applies different params to each element or whole batch

    A Mixup impl that's performed while collating the batches.
    """

    def _mix_elem_collate(self, output, batch, half=False):
        batch_size = len(batch)
        num_elem = batch_size // 2 if half else batch_size
        assert len(output) == num_elem
        lam_batch, use_cutmix = self._params_per_elem(num_elem)
        for i in range(num_elem):
            j = batch_size - i - 1
            lam = lam_batch[i]
            mixed = batch[i][0]
            if lam != 1.:
                if use_cutmix[i]:
                    if not half:
                        mixed = mixed.copy()
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        output.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    mixed[:, yl:yh, xl:xh] = batch[j][0][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    mixed = mixed.astype(np.float32) * lam + batch[j][0].astype(np.float32) * (1 - lam)
                    np.rint(mixed, out=mixed)
            output[i] += torch.from_numpy(mixed.astype(np.uint8))
        if half:
            lam_batch = np.concatenate((lam_batch, np.ones(num_elem)))
        return torch.tensor(lam_batch).unsqueeze(1)

    def _mix_pair_collate(self, output, batch):
        batch_size = len(batch)
        lam_batch, use_cutmix = self._params_per_elem(batch_size // 2)
        for i in range(batch_size // 2):
            j = batch_size - i - 1
            lam = lam_batch[i]
            mixed_i = batch[i][0]
            mixed_j = batch[j][0]
            assert 0 <= lam <= 1.0
            if lam < 1.:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        output.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    patch_i = mixed_i[:, yl:yh, xl:xh].copy()
                    mixed_i[:, yl:yh, xl:xh] = mixed_j[:, yl:yh, xl:xh]
                    mixed_j[:, yl:yh, xl:xh] = patch_i
                    lam_batch[i] = lam
                else:
                    mixed_temp = mixed_i.astype(np.float32) * lam + mixed_j.astype(np.float32) * (1 - lam)
                    mixed_j = mixed_j.astype(np.float32) * lam + mixed_i.astype(np.float32) * (1 - lam)
                    mixed_i = mixed_temp
                    np.rint(mixed_j, out=mixed_j)
                    np.rint(mixed_i, out=mixed_i)
            output[i] += torch.from_numpy(mixed_i.astype(np.uint8))
            output[j] += torch.from_numpy(mixed_j.astype(np.uint8))
        lam_batch = np.concatenate((lam_batch, lam_batch[::-1]))
        return torch.tensor(lam_batch).unsqueeze(1)

    def _mix_batch_collate(self, output, batch):
        batch_size = len(batch)
        lam, use_cutmix = self._params_per_batch()
        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                output.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
        for i in range(batch_size):
            j = batch_size - i - 1
            mixed = batch[i][0]
            if lam != 1.:
                if use_cutmix:
                    mixed = mixed.copy()  # don't want to modify the original while iterating
                    mixed[..., yl:yh, xl:xh] = batch[j][0][..., yl:yh, xl:xh]
                else:
                    mixed = mixed.astype(np.float32) * lam + batch[j][0].astype(np.float32) * (1 - lam)
                    np.rint(mixed, out=mixed)
            output[i] += torch.from_numpy(mixed.astype(np.uint8))
        return lam

    def __call__(self, batch, _=None):
        batch_size = len(batch)
        assert batch_size % 2 == 0, 'Batch size should be even when using this'
        half = 'half' in self.mode
        if half:
            batch_size //= 2
        output = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        if self.mode == 'elem' or self.mode == 'half':
            lam = self._mix_elem_collate(output, batch, half=half)
        elif self.mode == 'pair':
            lam = self._mix_pair_collate(output, batch)
        else:
            lam = self._mix_batch_collate(output, batch)
        target = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing, device='cpu')
        target = target[:batch_size]
        return output, target


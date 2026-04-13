# [CF] 2026-04-12:
# 这个文件包含了 VideoMAE 预训练的核心训练引擎。
# train_one_epoch 函数执行一个完整 epoch 的训练，包括：
# 1. 准备重建目标（labels）
# 2. 模型前向传播（只处理可见 patches）
# 3. 计算 MSE 损失
# 4. 反向传播和参数更新

import math
import sys
from typing import Iterable
import torch
import torch.nn as nn
import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None):
    """
    [CF] 执行一个 epoch 的 VideoMAE 预训练。
    
    Args:
        model: 预训练模型（PretrainVisionTransformer）
        data_loader: 数据加载器
        optimizer: 优化器
        device: 计算设备
        epoch: 当前 epoch 编号
        loss_scaler: 混合精度训练的梯度缩放器
        max_norm: 梯度裁剪的最大范数
        patch_size: 空间 patch 大小（通常为 16）
        normlize_target: 是否对重建目标做局部归一化
        log_writer: TensorBoard 日志写入器
        lr_scheduler: 学习率调度器
        start_steps: 起始步数（用于恢复训练）
        lr_schedule_values: 预计算的学习率数组
        wd_schedule_values: 预计算的权重衰减数组
    
    Returns:
        dict: 本 epoch 的各项指标平均值
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss()

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        # =====================================================================
        # [CF] 步骤 1：设置当前步的学习率和权重衰减
        # =====================================================================
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    # 学习率 = 基础调度值 × 层缩放因子（用于分层学习率）
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # =====================================================================
        # [CF] 步骤 2：获取数据并移至设备
        # =====================================================================
        videos, bool_masked_pos = batch
        videos = videos.to(device, non_blocking=True)
        # bool_masked_pos: [B, N] 布尔张量，True 表示该 patch 被掩码
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        # =====================================================================
        # [CF] 步骤 3：准备重建目标（Labels）
        # 这是 VideoMAE 的核心：模型要重建的是原始像素值
        # =====================================================================
        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
            unnorm_videos = videos * std + mean  # in [0, 1]

            if normlize_target:
                # 3.2a 对每个 patch 做局部归一化（推荐）
                # 目的：移除 patch 内的局部亮度和对比度差异，
                # 让模型专注于学习结构和运动模式
                
                # 将视频切分为 patches：形状变为 [B, num_patches, patch_pixel_dim, C]
                videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
                # 在每个 patch 内部计算均值和方差，进行归一化
                videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                    ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                # 将 patch 像素展平：形状变为 [B, num_patches, patch_pixel_dim * C]
                videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
            else:
                # 3.2b 不归一化，直接使用原始像素值
                videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)
            # 3.3 根据掩码提取被遮盖部分的 patches 作为标签
            B, _, C = videos_patch.shape
            labels = videos_patch[bool_masked_pos].reshape(B, -1, C)


        # =====================================================================
        # [CF] 步骤 4：模型前向传播 + 损失计算
        # =====================================================================
        with torch.cuda.amp.autocast(): # 混合精度：自动将部分计算转为 float16
            # 模型只接收可见 patches，输出对被遮盖 patches 的重建结果
            outputs = model(videos, bool_masked_pos)
            # outputs 形状: [B, num_masked_patches, patch_pixel_dim]
            loss = loss_func(input=outputs, target=labels)

        loss_value = loss.item()
        # 检查损失是否有效（非 NaN、非 Inf）
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # =====================================================================
        # [CF] 步骤 5：反向传播与参数更新
        # =====================================================================
        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        # 检查是否为二阶优化器（如 AdaHessian）
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        # 使用梯度缩放器执行反向传播和参数更新
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()
        # =====================================================================
        # [CF] 步骤 6：记录指标
        # =====================================================================
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        # 获取权重衰减值
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)
        # 写入 TensorBoard
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()
        # 步进学习率调度器
        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    # =========================================================================
    # [CF] Epoch 结束：汇总所有进程的指标并返回
    # =========================================================================
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

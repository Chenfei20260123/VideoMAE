# [CF] 2026-04-12:
# 这个文件是 VideoMAE 的工具箱，提供了训练过程中需要的各种辅助功能。
# 第一部分主要包含指标平滑统计和训练日志管理。

import io
import os
import math
import time
import json
from collections import defaultdict, deque
import datetime
import numpy as np
from timm.utils import get_state_dict
from torch.utils.data._utils.collate import default_collate
from pathlib import Path
import subprocess
import torch
import torch.distributed as dist
from torch._six import inf
import random

from tensorboardX import SummaryWriter


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    """
    [CF] 平滑值追踪器：用于追踪某个指标（如 loss）的近期值和全局平均值。
    
    它维护一个固定长度的滑动窗口，可以快速获取：
    - 窗口内的中位数、均值、最大值
    - 全局历史平均值
    这在训练过程中非常有用，可以平滑掉单步的波动，更清晰地观察趋势。
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """        
        """
        [CF] 在分布式训练中，同步所有进程的 count 和 total。
        
        注意：此方法不同步 deque（窗口内的历史值），
        因为每个进程的窗口内容不同，同步窗口没有意义。
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    """
    [CF] 指标日志记录器：管理多个 SmoothedValue，并提供优雅的迭代日志输出。
    
    它是训练循环中打印信息的核心工具，可以：
    - 追踪多个指标（loss, lr, grad_norm 等）
    - 计算 ETA（预计完成时间）
    - 监控数据加载时间和迭代时间
    - 显示 GPU 显存使用情况
    """
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """批量更新多个指标。"""
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """支持通过属性访问 meters，如 metric_logger.loss。"""
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        """生成所有指标的字符串表示。"""
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """同步所有进程的 meters（用于分布式训练）。"""
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """手动添加一个 meter。"""
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        [CF] 遍历迭代器，并按指定频率打印日志。
        
        这是训练循环中最常用的方法，它：
        1. 追踪迭代时间和数据加载时间
        2. 计算 ETA
        3. 监控 GPU 显存
        4. 按 print_freq 频率打印状态
        
        Args:
            iterable: 数据加载器
            print_freq: 打印频率（每多少步打印一次）
            header: 日志头（如 'Epoch: [0]'）
        
        Yields:
            迭代器的元素
        """
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        # 追踪迭代时间和数据加载时间的平滑值
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        # 用于格式化索引的宽度（保持对齐）
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        # 日志消息模板
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                # 计算预计剩余时间
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        # 打印整个 epoch 的总耗时
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

# [CF] ============================================================================
# [CF] 第二部分：TensorBoard 日志、分布式训练辅助、模型权重加载
# [CF] ============================================================================

class TensorboardLogger(object):
    """
    [CF] TensorBoard 日志记录器封装。
    
    将训练过程中的标量指标（loss, lr, grad_norm 等）写入 TensorBoard，
    便于在浏览器中可视化训练曲线。
    """
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        """设置当前步数。如果不指定，则自动递增。"""
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        """
        [CF] 批量记录多个标量指标。
        
        Args:
            head: 指标分组名称（如 'loss', 'opt'），在 TensorBoard 中会形成层级结构
            step: 指定步数，默认使用 self.step
            **kwargs: 指标名和值的键值对
        """
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        """强制将缓冲区数据写入磁盘。"""
        self.writer.flush()

def seed_worker(worker_id):
    """
    [CF] DataLoader 工作进程的随机种子初始化函数。
    
    确保每个数据加载工作进程有独立但可复现的随机状态。
    用于 DataLoader 的 worker_init_fn 参数。
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    """
    [CF] 为 ModelEma 加载检查点的适配函数。
    
    ModelEma 的 _load_checkpoint 方法期望接收一个文件路径，
    这个函数通过 BytesIO 将已加载的 checkpoint 对象模拟成文件，供其读取。
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """    
    """
    [CF] 设置分布式训练的打印行为：只有主进程可以打印。
    
    在分布式训练中，如果所有进程都打印日志，输出会非常混乱。
    此函数重写了内置的 print，非主进程的打印默认被抑制。
    如果需要强制打印（如报错信息），可以使用 print(..., force=True)。
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    """检查分布式训练环境是否可用且已初始化。"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """获取分布式训练的进程总数。"""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """获取当前进程的编号（0 到 world_size-1）。"""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """判断当前进程是否为主进程（rank == 0）。"""
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """只在主进程上保存模型，避免多进程同时写文件导致冲突。"""
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    """
    [CF] 初始化分布式训练模式。
    
    支持多种分布式启动方式：
    1. dist_on_itp: Intel 的分布式框架
    2. SLURM: 集群作业管理系统
    3. torch.distributed.launch / torchrun: PyTorch 标准方式
    
    该函数会解析环境变量，设置 rank、world_size、gpu 等参数，
    并初始化进程组。
    """
    if args.dist_on_itp:
        # Intel 分布式框架
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    elif 'SLURM_PROCID' in os.environ:
        # SLURM 集群环境
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = int(os.environ['SLURM_LOCALID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        os.environ['RANK'] = str(args.rank)
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['WORLD_SIZE'] = str(args.world_size)

        node_list = os.environ['SLURM_NODELIST']
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = addr
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # PyTorch 标准分布式启动（torch.distributed.launch 或 torchrun）
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        # 单机单卡模式
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    # assert torch.distributed.is_initialized()
    setup_for_distributed(args.rank == 0)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    """
    [CF] 灵活的模型权重加载函数。
    
    支持：
    1. 部分加载：允许预训练模型和当前模型结构不完全一致
    2. 忽略特定键：通过 ignore_missing 指定可以安全跳过的参数名模式
    3. 递归加载：支持嵌套模块的权重加载
    
    在 VideoMAE 中，这个函数用于：
    - 从预训练模型加载权重到微调模型（结构可能不同）
    - 从检查点恢复训练
    
    Args:
        model: 要加载权重的模型
        state_dict: 权重字典
        prefix: 模型名称前缀（用于递归调用）
        ignore_missing: 以 '|' 分隔的字符串，包含这些子串的缺失参数将被忽略
    
    Returns:
        None（直接修改 model）
    """
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

# [CF] ============================================================================
# [CF] 第三部分：混合精度训练、学习率调度、模型保存/恢复、DeepSpeed 配置
# [CF] ============================================================================
class NativeScalerWithGradNormCount:
    """
    [CF] PyTorch 混合精度训练的梯度缩放器封装。
    
    混合精度训练（AMP）使用 float16 进行前向和反向计算，可以：
    1. 大幅减少显存占用（约 40%）
    2. 加速计算（在 Tensor Core GPU 上）
    
    但 float16 的数值范围有限，梯度容易下溢（变为 0）。GradScaler 通过
    在前向时将 loss 放大，反向后再缩小梯度，来避免这个问题。
    """
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        """
        [CF] 执行一次训练步的梯度计算和参数更新。
        
        Args:
            loss: 损失值
            optimizer: 优化器
            clip_grad: 梯度裁剪阈值（None 表示不裁剪）
            parameters: 需要裁剪梯度的参数（clip_grad 不为 None 时必须提供）
            create_graph: 是否保留计算图（用于高阶梯度）
            update_grad: 是否更新参数（False 时只计算梯度不更新）
        
        Returns:
            norm: 梯度范数（如果 update_grad=True），否则 None
        """
        # 1. 缩放 loss 并反向传播
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                # 2a. 先将梯度从缩放状态恢复
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                # 2b. 裁剪梯度并计算范数
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                # 2a. 恢复梯度
                self._scaler.unscale_(optimizer)
                # 2b. 只计算梯度范数，不裁剪
                norm = get_grad_norm_(parameters)
            # 3. 更新参数（如果梯度有效）
            self._scaler.step(optimizer)
            # 4. 更新缩放因子
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        """获取缩放器的状态字典，用于保存检查点。"""
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        """从检查点恢复缩放器状态。"""
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    """
    [CF] 计算所有参数的梯度范数。
    
    支持 L1、L2（默认）、L∞ 范数。
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        # L∞ 范数：所有梯度绝对值的最大值
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        # Lp 范数：先计算每个参数梯度的 Lp 范数，再求整体的 Lp 范数
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    """
    [CF] 生成余弦衰减的学习率调度表。
    
    VideoMAE 使用 Warmup + Cosine Decay 的学习率策略：
    1. Warmup 阶段：学习率从 start_warmup_value 线性增长到 base_value
    2. Cosine Decay 阶段：学习率按余弦曲线从 base_value 衰减到 final_value
    
    Args:
        base_value: 基础学习率（Warmup 后的最大值）
        final_value: 最终学习率（衰减结束时的最小值）
        epochs: 总训练轮数
        niter_per_ep: 每个 epoch 的迭代步数
        warmup_epochs: Warmup 的 epoch 数
        start_warmup_value: Warmup 开始时的学习率
        warmup_steps: Warmup 的步数（如果 >0，则覆盖 warmup_epochs 的计算）
    
    Returns:
        np.ndarray: 长度为 epochs * niter_per_ep 的学习率数组
    """
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    # 生成 Warmup 阶段的学习率（线性增长）
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    # 生成 Cosine Decay 阶段的学习率
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    """
    [CF] 保存模型检查点。
    
    支持两种模式：
    1. 标准模式（有 loss_scaler）：保存完整状态（模型、优化器、epoch、scaler）
    2. DeepSpeed 模式：委托给 DeepSpeed 引擎保存
    """
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    """
    [CF] 自动加载模型检查点。
    
    支持两种模式：
    1. 标准模式：从 .pth 文件加载
    2. DeepSpeed 模式：委托给 DeepSpeed 引擎加载
    
    如果启用 --auto_resume，会自动查找 output_dir 下最新的检查点。
    """
    output_dir = Path(args.output_dir)
    if loss_scaler is not None:
        # 标准 PyTorch 训练模式
        # torch.amp
        if args.auto_resume and len(args.resume) == 0:
            # 自动查找最新的检查点
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            print("Auto resume checkpoint: %s" % args.resume)

        if args.resume:
            # 加载检查点
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            # 加载模型权重
            model_without_ddp.load_state_dict(checkpoint['model'])
            print("Resume checkpoint %s" % args.resume)
            # 如果检查点包含优化器和 epoch，则恢复训练状态
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch'] + 1
                if hasattr(args, 'model_ema') and args.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
    else:
        # deepspeed, only support '--auto_resume'.
        # DeepSpeed 模式，只支持 --auto_resume
        if args.auto_resume:
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
                args.start_epoch = client_states['epoch'] + 1
                if model_ema is not None:
                    if args.model_ema:
                        _load_checkpoint_for_ema(model_ema, client_states['model_ema'])


def create_ds_config(args):    
    """
    [CF] 生成 DeepSpeed 配置文件。
    
    DeepSpeed 是微软开发的深度学习优化库，支持：
    - ZeRO 优化（显存效率极高）
    - 混合精度训练
    - 梯度累积
    
    此函数生成一个基础的 DeepSpeed 配置，用于大规模分布式训练。
    """
    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-8
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128
            }
        }

        writer.write(json.dumps(ds_config, indent=2))

def multiple_samples_collate(batch, fold=False):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """    
    """
    [CF] 用于"重复增强"的数据收集函数。
    
    当对每个视频采样多个片段（num_sample > 1）时，每个样本会返回一个列表。
    此函数将这些嵌套列表展平，形成更大的批次。
    
    Args:
        batch: 原始批次数据
        fold: 是否将输入包裹在列表中
    
    Returns:
        整理后的批次数据
    """
    inputs, labels, video_idx, extra_data = zip(*batch)
    # 展平嵌套列表
    inputs = [item for sublist in inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    video_idx = [item for sublist in video_idx for item in sublist]
    # 使用默认的 collate 函数
    inputs, labels, video_idx, extra_data = (
        default_collate(inputs),
        default_collate(labels),
        default_collate(video_idx),
        default_collate(extra_data),
    )
    if fold:
        return [inputs], labels, video_idx, extra_data
    else:
        return inputs, labels, video_idx, extra_data

# [CF] 2026-04-12:
# 这个文件是 VideoMAE 的"优化器工厂"，负责根据配置参数创建优化器。
# 它包含两个核心功能:
# 1. 实现分层学习率衰减 (Layer-wise Learning Rate Decay)
# 2. 统一创建和管理各种优化器 (AdamW, SGD 等)

import torch
from torch import optim as optim

# 从 timm 库导入多种优化器实现
from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
from timm.optim.novograd import NovoGrad
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP

import json

# 尝试导入 NVIDIA Apex 库中的融合优化器（使用 CUDA 核融合技术加速）
try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def get_num_layer_for_vit(var_name, num_max_layer):
    """
    [CF] 根据参数名称，判断它属于 Vision Transformer 的哪一层。
    
    这是实现"分层学习率衰减"的辅助函数。为不同深度的层分配不同的学习率:
    - 浅层 (embedding 层): 学习率最小，因为它们提取的是通用低级特征
    - 深层 (靠近输出的层): 学习率较大，因为它们需要快速适应特定任务
    
    Args:
        var_name (str): 参数名称，如 "blocks.0.attn.qkv.weight"
        num_max_layer (int): 模型的总层数（深度）
    
    Returns:
        int: 该参数所属的层索引 (0 表示最浅层，num_max_layer-1 表示最深层)
    """
    # 特殊的全局参数，视为第 0 层（最浅层）
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    # Patch Embedding 层，也视为第 0 层
    elif var_name.startswith("patch_embed"):
        return 0
    # 相对位置偏置，通常放在较深层
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    # Transformer Block 中的参数，从名称中解析层号
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    # 其他参数（如分类头）视为最深层
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    """
    [CF] 分层学习率衰减的"分配器"。
    
    它存储了每一层的学习率缩放因子，并根据参数名称返回对应的缩放值。
    典型的衰减策略: lr_layer = base_lr * (decay_rate)^(depth - layer_id)
    """
    def __init__(self, values):
        """
        Args:
            values (list): 长度为 num_layers 的列表，每个元素是该层的学习率缩放因子
        """
        self.values = values

    def get_scale(self, layer_id):
        """根据层索引返回学习率缩放因子"""
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        """根据参数名称返回其所属的层索引"""
        return get_num_layer_for_vit(var_name, len(self.values))


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    """
    [CF] 将模型的参数分组，以便为不同组设置不同的学习率和权重衰减。
    
    这是实现"分层学习率"和"差异化权重衰减"的核心函数。
    它遍历模型的所有参数，根据规则将其分配到不同的参数组中。
    
    Args:
        model: 模型实例
        weight_decay (float): 权重衰减系数
        skip_list (tuple): 跳过权重衰减的参数名称列表
        get_num_layer (callable): 函数，根据参数名返回层索引
        get_layer_scale (callable): 函数，根据层索引返回学习率缩放因子
    
    Returns:
        list: 参数组列表，每个元素是一个字典，包含 'params', 'weight_decay', 'lr_scale' 等
    """
    parameter_group_names = {} # 用于打印的参数组信息（存储参数名）
    parameter_group_vars = {} # 实际返回的参数组（存储参数张量）

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights # 跳过冻结的参数
        # [CF] 规则1: 判断是否应用权重衰减
        # 对于 1 维参数（如 LayerNorm 的 weight/bias）、偏置项、或 skip_list 中的参数，不应用权重衰减
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        # [CF] 规则2: 如果提供了分层函数，计算参数所属的层，并按层进一步分组
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        # [CF] 如果是新的分组，创建对应的条目
        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
        # [CF] 将参数添加到对应的组中
        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None):
    """
    [CF] 创建优化器的工厂函数。
    
    根据命令行参数 args.opt 选择优化器类型，并配置相应的参数。
    支持 PyTorch 原生优化器、timm 库中的优化器、以及 NVIDIA Apex 融合优化器。
    
    Args:
        args: 命令行参数对象，包含 opt, lr, weight_decay, opt_eps, opt_betas, momentum 等
        model: 要优化的模型
        get_num_layer: 分层函数
        get_layer_scale: 层缩放因子获取函数
        filter_bias_and_bn (bool): 是否过滤偏置和 BN 的权重衰减
        skip_list: 额外跳过权重衰减的参数列表
    
    Returns:
        torch.optim.Optimizer: 配置好的优化器实例
    """
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    # [CF] 步骤1: 构建参数组
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        # 调用 get_parameter_groups 进行详细分组
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale)
        weight_decay = 0. # 分组后，优化器级别的 weight_decay 设为 0，因为每个组已独立设置
    else:
        parameters = model.parameters()
        
    # [CF] 步骤2: 检查融合优化器的前提条件
    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    print("optimizer settings:", opt_args)

    # [CF] 步骤4: 根据优化器名称创建对应的优化器实例
    # 处理 'lookahead_sgd' 这种复合名称（提取最后一部分）
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]

    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'novograd':
        optimizer = NovoGrad(parameters, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    # [CF] Apex 融合优化器（使用 CUDA 核融合，更快更省显存）
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError
    # [CF] 步骤5: 如果指定了 Lookahead，用 Lookahead 包装基础优化器
    # Lookahead 是一种"前瞻"优化技巧，可以稳定训练、加速收敛
    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer

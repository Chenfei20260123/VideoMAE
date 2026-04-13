# [CF] 2026-04-12:
# 这个文件定义了 VideoMAE 预训练专用的模型架构。
# 与 modeling_finetune.py 中的标准 ViT 不同，预训练模型包含：
# 1. 编码器：只处理可见 patches（约 10%）
# 2. 解码器：从编码特征和掩码令牌重建被遮盖的 patches
# 3. 掩码令牌：可学习的向量，代表被遮盖的 patches

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import partial

from modeling_finetune import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_



def trunc_normal_(tensor, mean=0., std=1.):
    """
    [CF] 截断正态分布初始化。
    将 timm 的 trunc_normal_ 封装，默认在 [-std, std] 范围内采样。
    """
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    'pretrain_videomae_small_patch16_224',
    'pretrain_videomae_base_patch16_224', 
    'pretrain_videomae_large_patch16_224', 
    'pretrain_videomae_huge_patch16_224',
]


class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    """
    [CF] VideoMAE 预训练专用的 ViT 编码器。
    
    与标准编码器的核心区别：
    在 forward_features 中，通过 x[~mask] 操作，只保留可见 patches，
    丢弃被遮盖的 patches。这使得编码器的计算量约为原来的 10%（当掩码率=0.9时）。
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2, use_checkpoint=False,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # 1. Patch Embedding：将视频转换为 patch 序列
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint


        # TODO: Add the cls token
        # 2. 位置编码
        if use_learnable_pos_emb:
            # 可学习的位置编码（+1 是为了预留 CLS token，虽然这里没用到）
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # 正弦-余弦位置编码（固定，不参与训练）
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        # 3. Transformer Blocks
        # DropPath 率随深度线性增加
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        # 4. 最后的 LayerNorm
        self.norm =  norm_layer(embed_dim)
        # 5. 分类头（预训练时 num_classes=0，所以是 Identity）
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        # 初始化可学习的位置编码
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        """权重初始化函数。"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        """指定不应用权重衰减的参数。"""
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask):
        """
        [CF] 提取特征的核心函数。
        
        Args:
            x: 输入视频 [B, C, T, H, W]
            mask: 布尔掩码 [B, N]，True 表示被遮盖
        
        Returns:
            x_vis: 可见 patches 的编码特征 [B, N_vis, embed_dim]
        """
        _, _, T, _, _ = x.shape
        # 1. Patch Embedding: [B, C, T, H, W] -> [B, N, embed_dim]
        x = self.patch_embed(x)
        # 2. 添加位置编码
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        # 3. [CF] 关键操作：只保留可见 patches！
        # ~mask 取反：False 表示可见，True 表示被遮盖
        # x[~mask] 会返回一个一维张量，然后再 reshape 回 [B, N_vis, C]
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible
        # 4. 通过 Transformer Blocks
        if self.use_checkpoint:
            # 使用梯度检查点：牺牲计算时间换取显存
            for blk in self.blocks:
                x_vis = checkpoint.checkpoint(blk, x_vis)
        else:   
            for blk in self.blocks:
                x_vis = blk(x_vis)

        # 5. 最后的 LayerNorm
        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x

# [CF] ============================================================================
# [CF] 第二部分：预训练解码器 与 完整的 VideoMAE 预训练模型
# [CF] ============================================================================
class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """    
    """
    [CF] VideoMAE 预训练专用的 ViT 解码器。
    
    解码器的任务是：接收一个完整的 patch 序列（包含可见 patches 的编码特征 
    和掩码令牌），通过 Transformer 层处理后，预测被遮盖 patches 的原始像素值。
    
    关键特点：
    1. 轻量化：通常比编码器更窄、更浅
    2. 只在末尾输出被遮盖部分的预测
    """
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=None, num_patches=196, tubelet_size=2, use_checkpoint=False
                 ):
        super().__init__()
        self.num_classes = num_classes
        # [CF] 验证输出维度：每个 tube 的像素总数
        # 对于 tubelet_size=2, patch_size=16: 3 * 2 * 16 * 16 = 1536
        assert num_classes == 3 * tubelet_size * patch_size ** 2 
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint

        # Transformer Blocks（通常比编码器少）
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        # 输出头：将特征投影到像素值
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        """
        [CF] 解码器前向传播。
        
        Args:
            x: 完整的序列 [B, N, embed_dim]（可见特征 + 掩码令牌）
            return_token_num: 需要返回的掩码令牌数量（即被遮盖的 patch 数）
        
        Returns:
            被遮盖部分的像素预测 [B, return_token_num, num_classes]
        """
        # 通过 Transformer Blocks
        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:   
            for blk in self.blocks:
                x = blk(x)

        # [CF] 关键：只取序列末尾的掩码令牌进行预测
        # 因为构建序列时，掩码令牌被拼接在末尾
        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x

class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    """
    [CF] VideoMAE 完整的预训练模型。
    
    组装了编码器、解码器、掩码令牌和位置编码，实现了完整的掩码重建流程。
    
    架构特点（非对称设计）：
    1. 编码器：深而宽，但只处理 ~10% 的可见 patches
    2. 解码器：浅而窄，处理完整的 patch 序列
    3. 编码器-解码器连接：通过线性层将编码特征映射到解码器维度
    4. 掩码令牌：可学习的向量，代表被遮盖的 patches
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 encoder_in_chans=3, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=768, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=1536, #  decoder_num_classes=768, 
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 use_checkpoint=False,
                 tubelet_size=2,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 ):
        super().__init__()
        # 1. 编码器：只处理可见 patches
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            use_learnable_pos_emb=use_learnable_pos_emb)
        
        # 2. 解码器：处理完整序列，预测被遮盖像素
        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size, 
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint)

        # 3. 编码器到解码器的特征维度映射
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        # 4. 掩码令牌：可学习的向量，代表被遮盖的 patch
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # 5. 解码器的位置编码（完整的 patch 序列）
        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)
        # 初始化掩码令牌
        trunc_normal_(self.mask_token, std=.02)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask):
        """
        [CF] VideoMAE 完整的前向传播。
        
        Args:
            x: 输入视频 [B, C, T, H, W]
            mask: 布尔掩码 [B, N]，True 表示被遮盖
        
        Returns:
            被遮盖 patches 的重建像素值 [B, N_mask, 3 * tubelet * patch^2]
        """
        _, _, T, _, _ = x.shape
        # =====================================================================
        # 阶段 1：编码
        # =====================================================================
        # 编码器只处理可见 patches
        # x_vis 形状: [B, N_vis, encoder_embed_dim]（如 [B, 157, 768]）
        x_vis = self.encoder(x, mask) # [B, N_vis, C_e]
        # 映射到解码器维度
        # x_vis 形状: [B, N_vis, decoder_embed_dim]（如 [B, 157, 384]）
        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order, 
        # but shuffle the pos embedding accorddingly.
        # =====================================================================
        # 阶段 2：组装解码器输入序列
        # =====================================================================
        # 获取完整的位置编码并复制到批次中的每个样本
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        # 根据掩码分离可见部分和遮盖部分的位置编码
        # ~mask: 可见位置
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        # mask: 遮盖位置
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        # [CF] 关键：拼接完整序列
        # 1. 可见部分：编码特征 + 对应的位置编码
        # 2. 遮盖部分：掩码令牌 + 对应的位置编码
        # 拼接后形状: [B, N_total, decoder_embed_dim]
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # [B, N, C_d]
        # =====================================================================
        # 阶段 3：解码与重建
        # =====================================================================
        # 解码器处理完整序列，并只返回遮盖部分的预测
        # 输出形状: [B, N_mask, 3 * tubelet * patch^2]（如 [B, 1411, 1536]）
        x = self.decoder(x_full, pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16]

        return x

# [CF] ============================================================================
# [CF] 第三部分：模型注册与工厂函数
# [CF] 定义了四种不同规模的 VideoMAE 预训练模型：Small, Base, Large, Huge
# [CF] ============================================================================
@register_model
def pretrain_videomae_small_patch16_224(pretrained=False, **kwargs):
    """
    [CF] VideoMAE-Small 预训练模型。
    
    最小的 VideoMAE 变体，适合资源有限的场景或快速实验。
    
    关键配置：
    - 编码器: embed_dim=384, depth=12, num_heads=6 (约 22M 参数)
    - 解码器: embed_dim=192, depth=4,  num_heads=3
    """
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=1536, 
        decoder_embed_dim=192, 
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def pretrain_videomae_base_patch16_224(pretrained=False, **kwargs):
    """
    [CF] VideoMAE-Base 预训练模型。
    
    VideoMAE 最常用的配置，在效果和效率之间取得了良好平衡。
    
    关键配置：
    - 编码器: embed_dim=768, depth=12, num_heads=12 (约 86M 参数)
    - 解码器: embed_dim=384, depth=4,  num_heads=6
    
    注意：解码器的 depth 没有显式指定，使用 PretrainVisionTransformerDecoder 
    的默认值（通常为 4 或从 kwargs 传入）。
    """
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=768, 
        encoder_depth=12, 
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
 
@register_model
def pretrain_videomae_large_patch16_224(pretrained=False, **kwargs):
    """
    [CF] VideoMAE-Large 预训练模型。
    
    更大规模的模型，适合有充足计算资源的场景，追求更高的精度。
    
    关键配置：
    - 编码器: embed_dim=1024, depth=24, num_heads=16 (约 307M 参数)
    - 解码器: embed_dim=512,  depth=4,  num_heads=8
    """
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=1024, 
        encoder_depth=24, 
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=1536, 
        decoder_embed_dim=512,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def pretrain_videomae_huge_patch16_224(pretrained=False, **kwargs):
    """
    [CF] VideoMAE-Huge 预训练模型。
    
    最大规模的 VideoMAE 变体，需要大量计算资源（多卡训练）。
    
    关键配置：
    - 编码器: embed_dim=1280, depth=32, num_heads=16 (约 632M 参数)
    - 解码器: embed_dim=640,  depth=4,  num_heads=8
    """
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=1280, 
        encoder_depth=32, 
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=1536, 
        decoder_embed_dim=640,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

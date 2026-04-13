# -*- coding: utf-8 -*-
# [CF] 2026-04-13:
# 这个脚本用于可视化 VideoMAE 预训练模型的重建效果。
# 它加载一个训练好的模型，对输入视频应用掩码，然后重建被遮盖的部分，
# 并保存原始帧、掩码帧和重建帧的图像。
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from pathlib import Path
from timm.models import create_model
import utils
import modeling_pretrain
from datasets import DataAugmentationForVideoMAE
from torchvision.transforms import ToPILImage
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from decord import VideoReader, cpu
from torchvision import transforms
from transforms import *
from masking_generator import  TubeMaskingGenerator

class DataAugmentationForVideoMAE(object):
    """
    [CF] 为可视化定制的数据增强类。
    与训练时的版本不同，这里使用中心裁剪（GroupCenterCrop）而非多尺度随机裁剪，
    以保证可视化结果的确定性和一致性。
    """
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406] # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225] # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        # 使用中心裁剪，确保可视化结果稳定
        self.train_augmentation = GroupCenterCrop(args.input_size)
        self.transform = transforms.Compose([                            
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data , _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

def get_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser('VideoMAE visualization reconstruction script', add_help=False)
    parser.add_argument('img_path', type=str, help='input video path')
    parser.add_argument('save_path', type=str, help='save video path')
    parser.add_argument('model_path', type=str, help='checkpoint path of model')
    parser.add_argument('--mask_type', default='random', choices=['random', 'tube'],
                        type=str, help='masked strategy of video tokens/patches')
    parser.add_argument('--num_frames', type=int, default= 16)
    parser.add_argument('--sampling_rate', type=int, default= 4)
    parser.add_argument('--decoder_depth', default=4, type=int,
                        help='depth of decoder')
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size for backbone')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    # Model parameters
    parser.add_argument('--model', default='pretrain_videomae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    return parser.parse_args()


def get_model(args):
    """加载预训练模型。"""
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        decoder_depth=args.decoder_depth
    )

    return model


def main(args):
    print(args)

    device = torch.device(args.device)
    cudnn.benchmark = True
    # 创建模型并加载权重
    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    if args.save_path:
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

    # 加载视频并采样帧
    with open(args.img_path, 'rb') as f:
        vr = VideoReader(f, ctx=cpu(0))
    duration = len(vr)
    new_length  = 1 
    new_step = 1
    skip_length = new_length * new_step
    # frame_id_list = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61]

    # 选择要可视化的帧索引（这里硬编码了从第60帧开始，每隔一帧取一帧，共16帧）
    tmp = np.arange(0,32, 2) + 60
    frame_id_list = tmp.tolist()
    # average_duration = (duration - skip_length + 1) // args.num_frames
    # if average_duration > 0:
    #     frame_id_list = np.multiply(list(range(args.num_frames)),
    #                             average_duration)
    #     frame_id_list = frame_id_list + np.random.randint(average_duration,
    #                                             size=args.num_frames)

    video_data = vr.get_batch(frame_id_list).asnumpy()
    print(video_data.shape)
    img = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
    # 应用数据增强和掩码生成
    transforms = DataAugmentationForVideoMAE(args)
    img, bool_masked_pos = transforms((img, None)) # T*C,H,W
    # print(img.shape)
    img = img.view((args.num_frames , 3) + img.size()[-2:]).transpose(0,1) # T*C,H,W -> T,C,H,W -> C,T,H,W
    # img = img.view(( -1 , args.num_frames) + img.size()[-2:]) 
    bool_masked_pos = torch.from_numpy(bool_masked_pos)

    with torch.no_grad():
        # img = img[None, :]
        # bool_masked_pos = bool_masked_pos[None, :]
        img = img.unsqueeze(0)
        print(img.shape)
        bool_masked_pos = bool_masked_pos.unsqueeze(0)
        
        img = img.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        # 模型前向传播，获取重建结果
        outputs = model(img, bool_masked_pos)

        #save original video # ===== 保存原始视频帧 =====
        mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
        std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
        ori_img = img * std + mean  # in [0, 1]
        imgs = [ToPILImage()(ori_img[0,:,vid,:,:].cpu()) for vid, _ in enumerate(frame_id_list)  ]
        for id, im in enumerate(imgs):
            im.save(f"{args.save_path}/ori_img{id}.jpg")

        # ===== 重建被遮盖的 patches =====
        # 将视频切分为 patches，并进行局部归一化（与训练时一致）
        img_squeeze = rearrange(ori_img, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size[0], p2=patch_size[0])
        img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        # 用模型输出替换被遮盖的部分
        img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
        img_patch[bool_masked_pos] = outputs

        #make mask # ===== 生成掩码可视化 =====
        mask = torch.ones_like(img_patch)
        mask[bool_masked_pos] = 0
        mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
        mask = rearrange(mask, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2) ', p0=2, p1=patch_size[0], p2=patch_size[1], h=14, w=14)

        #save reconstruction video # ===== 保存重建视频帧 =====
        rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
        # Notice: To visualize the reconstruction video, we add the predict and the original mean and var of each patch.
        # 反归一化：加回均值和方差
        rec_img = rec_img * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + img_squeeze.mean(dim=-2, keepdim=True)
        rec_img = rearrange(rec_img, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)', p0=2, p1=patch_size[0], p2=patch_size[1], h=14, w=14)
        imgs = [ ToPILImage()(rec_img[0, :, vid, :, :].cpu().clamp(0,0.996)) for vid, _ in enumerate(frame_id_list)  ]

        for id, im in enumerate(imgs):
            im.save(f"{args.save_path}/rec_img{id}.jpg")

        #save masked video  # ===== 保存掩码视频帧（只显示被遮盖的部分） =====
        img_mask = rec_img * mask
        imgs = [ToPILImage()(img_mask[0, :, vid, :, :].cpu()) for vid, _ in enumerate(frame_id_list)]
        for id, im in enumerate(imgs):
            im.save(f"{args.save_path}/mask_img{id}.jpg")

if __name__ == '__main__':
    opts = get_args()
    main(opts)

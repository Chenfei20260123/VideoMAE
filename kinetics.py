# [CF] 2026-04-13:
# 这个文件定义了用于视频分类任务的数据集类。
# VideoClsDataset: 用于 Kinetics-400, UCF101, HMDB51 等标准数据集
# VideoMAE: 专门用于 VideoMAE 预训练的数据集类

import os
import numpy as np
from numpy.lib.function_base import disp
import torch
import decord
from PIL import Image
from torchvision import transforms
from random_erasing import RandomErasing
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset
import video_transforms as video_transforms 
import volume_transforms as volume_transforms

class VideoClsDataset(Dataset):
    """Load your own video classification dataset."""    
    """
    [CF] 通用视频分类数据集类。
    
    用于加载 Kinetics-400, UCF101, HMDB51 等标准视频分类数据集。
    支持三种模式：
    - 'train': 训练模式，应用数据增强
    - 'validation': 验证模式，使用中心裁剪
    - 'test': 测试模式，使用多视角（时间片段+空间裁剪）评估
    """

    def __init__(self, anno_path, data_path, mode='train', clip_len=8,
                 frame_sample_rate=2, crop_size=224, short_side_size=256,
                 new_height=256, new_width=340, keep_aspect_ratio=True,
                 num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3,args=None):
        self.anno_path = anno_path          # 标注文件路径（CSV格式）
        self.data_path = data_path          # 数据根目录
        self.mode = mode                    # 模式：train / validation / test
        self.clip_len = clip_len            # 采样的帧数
        self.frame_sample_rate = frame_sample_rate  # 采样步长（每隔几帧取一帧）
        self.crop_size = crop_size          # 裁剪尺寸
        self.short_side_size = short_side_size  # 短边尺寸（用于缩放）
        self.new_height = new_height        # 固定高度（如果不保持长宽比）
        self.new_width = new_width          # 固定宽度（如果不保持长宽比）
        self.keep_aspect_ratio = keep_aspect_ratio  # 是否保持长宽比
        self.num_segment = num_segment      # 训练时的时间片段数
        self.test_num_segment = test_num_segment  # 测试时的时间片段数
        self.num_crop = num_crop            # 训练时的空间裁剪数
        self.test_num_crop = test_num_crop  # 测试时的空间裁剪数
        self.args = args
        # 是否启用数据增强和随机擦除
        self.aug = False
        self.rand_erase = False
        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True
        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")
        
        # 读取标注文件：格式为 "视频路径 标签"
        import pandas as pd
        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=' ')
        self.dataset_samples = list(cleaned.values[:, 0])
        self.label_array = list(cleaned.values[:, 1])

        if (mode == 'train'):
            pass

        elif (mode == 'validation'):
            # 验证模式：缩放到短边 -> 中心裁剪 -> 转张量 -> 标准化
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(self.short_side_size, interpolation='bilinear'),
                video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            # 测试模式：先只做缩放，后续手动进行多视角裁剪
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(size=(short_side_size), interpolation='bilinear')
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
            # 构建测试用的多视角索引
            # 每个视频会被采样 test_num_segment × test_num_crop 个视角
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        sample_label = self.label_array[idx]
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_seg.append((ck, cp))

    def __getitem__(self, index):
        """获取一个训练/验证/测试样本。"""
        if self.mode == 'train':
            args = self.args 
            scale_t = 1

            sample = self.dataset_samples[index]
            # 加载视频帧：返回 numpy 数组，形状为 (T, H, W, C)
            buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t) # T H W C
            if len(buffer) == 0:
                # 如果视频加载失败，随机选择另一个视频
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t)

            if args.num_sample > 1:
                # 重复采样：对同一个视频采样多个增强版本（用于测试时的多视角评估）
                frame_list = []
                label_list = []
                index_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self.label_array[index]
                    frame_list.append(new_frames)
                    label_list.append(label)
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                buffer = self._aug_frame(buffer, args)

            # 返回：视频张量, 标签, 视频文件名
            return buffer, self.label_array[index], index, {}

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            buffer = self.loadvideo_decord(sample)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample)
            buffer = self.data_transform(buffer)
            return buffer, self.label_array[index], sample.split("/")[-1].split(".")[0]

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.loadvideo_decord(sample)

            while len(buffer) == 0:
                warnings.warn("video {}, temporal {}, spatial {} not found during testing".format(\
                    str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer = self.loadvideo_decord(sample)

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                                 / (self.test_num_crop - 1)
            temporal_step = max(1.0 * (buffer.shape[0] - self.clip_len) \
                                / (self.test_num_segment - 1), 0)
            temporal_start = int(chunk_nb * temporal_step)
            spatial_start = int(split_nb * spatial_step)
            if buffer.shape[1] >= buffer.shape[2]:
                buffer = buffer[temporal_start:temporal_start + self.clip_len, \
                       spatial_start:spatial_start + self.short_side_size, :, :]
            else:
                buffer = buffer[temporal_start:temporal_start + self.clip_len, \
                       :, spatial_start:spatial_start + self.short_side_size, :]

            buffer = self.data_transform(buffer)
            return buffer, self.test_label_array[index], sample.split("/")[-1].split(".")[0], \
                   chunk_nb, split_nb
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(
        self,
        buffer,
        args,
    ):
        """
        [CF] 对加载的视频帧应用数据增强。
        
        增强流程：
        1. 创建 RandAugment 变换
        2. 转换为 PIL 图像并应用 RandAugment
        3. 转回张量并调整维度顺序
        4. 标准化
        5. 空间采样（随机裁剪、翻转等）
        6. 随机擦除（可选）
        """

        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [
            transforms.ToPILImage()(frame) for frame in buffer
        ]

        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer) # T C H W
        buffer = buffer.permute(0, 2, 3, 1) # T H W C 
        
        # T H W C 
        buffer = tensor_normalize(
            buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=False if args.data_set == 'SSV2' else True ,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer


    def loadvideo_decord(self, sample, sample_rate_scale=1):
        """Load video content using Decord"""        
        """
        [CF] 使用 Decord 库加载视频帧。
        
        支持两种采样策略：
        - 训练/验证：将视频均匀分段，每段内随机采样
        - 测试：均匀采样整个视频
        
        Args:
            sample: 视频文件路径
            sample_rate_scale: 采样率缩放因子
        
        Returns:
            numpy 数组，形状为 (T, H, W, C)
        """
        fname = sample

        if not (os.path.exists(fname)):
            return []

        # avoid hanging issue
        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []
        try:
            if self.keep_aspect_ratio:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(fname, width=self.new_width, height=self.new_height,
                                 num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

        if self.mode == 'test':
            all_index = [x for x in range(0, len(vr), self.frame_sample_rate)]
            while len(all_index) < self.clip_len:
                all_index.append(all_index[-1])
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer

        # handle temporal segments
        converted_len = int(self.clip_len * self.frame_sample_rate)
        seg_len = len(vr) // self.num_segment

        all_index = []
        for i in range(self.num_segment):
            if seg_len <= converted_len:
                index = np.linspace(0, seg_len, num=seg_len // self.frame_sample_rate)
                index = np.concatenate((index, np.ones(self.clip_len - seg_len // self.frame_sample_rate) * seg_len))
                index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                end_idx = np.random.randint(converted_len, seg_len)
                str_idx = end_idx - converted_len
                index = np.linspace(str_idx, end_idx, num=self.clip_len)
                index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
            index = index + i*seg_len
            all_index.extend(list(index))

        all_index = all_index[::int(sample_rate_scale)]
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """    
    """
    [CF] 对视频帧进行空间采样（缩放、裁剪、翻转）。
    
    这是训练/测试阶段的核心空间变换函数，支持两种模式：
    1. 训练模式 (spatial_idx=-1)：随机缩放 + 随机裁剪 + 随机翻转
    2. 测试模式 (spatial_idx=0,1,2)：固定缩放 + 均匀裁剪（左/中/右 或 上/中/下）
    
    Args:
        frames (tensor): 视频帧，形状为 (T, H, W, C)
        spatial_idx (int): -1 为随机采样；0,1,2 为均匀采样
        min_scale, max_scale (int): 短边缩放范围
        crop_size (int): 裁剪尺寸
        random_horizontal_flip (bool): 是否随机水平翻转
        aspect_ratio (list): 宽高比范围，如 [0.75, 1.3333]
        scale (list): 面积比例范围，如 [0.08, 1.0]
        motion_shift (bool): 是否使用带位移的裁剪（模拟相机运动）
    
    Returns:
        frames (tensor): 空间采样后的视频帧
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        # ===== 训练模式：随机空间采样 =====
        if aspect_ratio is None and scale is None:
            # 传统方式：短边缩放 + 随机裁剪
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            # Inception 风格：随机缩放裁剪（可选带位移）
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        # ===== 测试模式：确定性空间采样 =====
        # 测试时不应有随机性，min_scale, max_scale, crop_size 应相等
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """    
    """
    [CF] 张量标准化：(tensor - mean) / std。
    
    支持 uint8 输入（自动转为 float 并除以 255）。
    
    Args:
        tensor (tensor): 待标准化的张量
        mean (list): 均值
        std (list): 标准差
    
    Returns:
        tensor: 标准化后的张量
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


class VideoMAE(torch.utils.data.Dataset):
    """Load your own video classification dataset.
    Parameters
    ----------
    root : str, required.
        Path to the root folder storing the dataset.
    setting : str, required.
        A text file describing the dataset, each line per video sample.
        There are three items in each line: (1) video path; (2) video length and (3) video label.
    train : bool, default True.
        Whether to load the training or validation set.
    test_mode : bool, default False.
        Whether to perform evaluation on the test set.
        Usually there is three-crop or ten-crop evaluation strategy involved.
    name_pattern : str, default None.
        The naming pattern of the decoded video frames.
        For example, img_00012.jpg.
    video_ext : str, default 'mp4'.
        If video_loader is set to True, please specify the video format accordinly.
    is_color : bool, default True.
        Whether the loaded image is color or grayscale.
    modality : str, default 'rgb'.
        Input modalities, we support only rgb video frames for now.
        Will add support for rgb difference image and optical flow image later.
    num_segments : int, default 1.
        Number of segments to evenly divide the video into clips.
        A useful technique to obtain global video-level information.
        Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016.
    num_crop : int, default 1.
        Number of crops for each image. default is 1.
        Common choices are three crops and ten crops during evaluation.
    new_length : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_step : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    temporal_jitter : bool, default False.
        Whether to temporally jitter if new_step > 1.
    video_loader : bool, default False.
        Whether to use video loader to load data.
    use_decord : bool, default True.
        Whether to use Decord video loader to load data. Otherwise use mmcv video loader.
    transform : function, default None.
        A function that takes data and label and transforms them.
    data_aug : str, default 'v1'.
        Different types of data augmentation auto. Supports v1, v2, v3 and v4.
    lazy_init : bool, default False.
        If set to True, build a dataset instance without loading any dataset.
    """    
    """
    [CF] 专门为 VideoMAE 预训练设计的数据集类。
    
    与 VideoClsDataset 的关键区别：
    1. 不返回标签，而是返回 (视频张量, 掩码)
    2. 使用 DataAugmentationForVideoMAE 作为 transform，它会在内部生成管道掩码
    3. 时间采样策略：均匀分段 + 每段内连续采样
    
    参数说明（主要）:
        root: 数据根目录
        setting: 标注文件路径，每行格式: "视频路径 标签"
        new_length: 采样的帧数（如 16）
        new_step: 采样步长（如 2，表示每隔一帧取一帧）
        transform: 数据增强函数（通常是 DataAugmentationForVideoMAE 实例）
    """
    def __init__(self,
                 root,
                 setting,
                 train=True,
                 test_mode=False,
                 name_pattern='img_%05d.jpg',
                 video_ext='mp4',
                 is_color=True,
                 modality='rgb',
                 num_segments=1,
                 num_crop=1,
                 new_length=1,
                 new_step=1,
                 transform=None,
                 temporal_jitter=False,
                 video_loader=False,
                 use_decord=False,
                 lazy_init=False):

        super(VideoMAE, self).__init__()
        self.root = root
        self.setting = setting
        self.train = train
        self.test_mode = test_mode
        self.is_color = is_color
        self.modality = modality
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.temporal_jitter = temporal_jitter
        self.name_pattern = name_pattern
        self.video_loader = video_loader
        self.video_ext = video_ext
        self.use_decord = use_decord
        self.transform = transform
        self.lazy_init = lazy_init


        if not self.lazy_init:
            self.clips = self._make_dataset(root, setting)
            if len(self.clips) == 0:
                raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                                   "Check your data directory (opt.data-dir)."))

    def __getitem__(self, index):
        """
        [CF] 获取一个预训练样本。
        
        Returns:
            process_data: 视频张量，形状为 (C, T, H, W)
            mask: 管道掩码，形状为 (N,) 的布尔数组
        """
        directory, target = self.clips[index]
        if self.video_loader:
            if '.' in directory.split('/')[-1]:
                # data in the "setting" file already have extension, e.g., demo.mp4
                video_name = directory
            else:
                # data in the "setting" file do not have extension, e.g., demo
                # So we need to provide extension (i.e., .mp4) to complete the file name.
                video_name = '{}.{}'.format(directory, self.video_ext)

            decord_vr = decord.VideoReader(video_name, num_threads=1)
            duration = len(decord_vr)

        segment_indices, skip_offsets = self._sample_train_indices(duration)

        images = self._video_TSN_decord_batch_loader(directory, decord_vr, duration, segment_indices, skip_offsets)

        process_data, mask = self.transform((images, None)) # T*C,H,W
        process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
        
        return (process_data, mask)

    def __len__(self):
        return len(self.clips)

    def _make_dataset(self, directory, setting):
        """
        [CF] 读取标注文件，构建样本列表。
        
        标注文件格式：每行 "视频路径 标签"
        """
        if not os.path.exists(setting):
            raise(RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (setting)))
        clips = []
        with open(setting) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split(' ')
                # line format: video_path, video_duration, video_label
                if len(line_info) < 2:
                    raise(RuntimeError('Video input format is not correct, missing one or more element. %s' % line))
                clip_path = os.path.join(line_info[0])
                target = int(line_info[1])
                item = (clip_path, target)
                clips.append(item)
        return clips

    def _sample_train_indices(self, num_frames):
        """
        [CF] 采样训练帧索引（均匀分段 + 每段内连续采样）。
        
        这是 VideoMAE 时间采样的核心：
        1. 将视频均匀分成 num_segments 段
        2. 在每段内随机选择一个起始点
        3. 从起始点开始，以 new_step 为步长，连续取 new_length 帧
        
        Returns:
            offsets: 每段的起始帧索引
            skip_offsets: 段内的偏移量（用于 temporal jitter）
        """
        average_duration = (num_frames - self.skip_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)),
                                  average_duration)
            offsets = offsets + np.random.randint(average_duration,
                                                  size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(np.random.randint(
                num_frames - self.skip_length + 1,
                size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets


    def _video_TSN_decord_batch_loader(self, directory, video_reader, duration, indices, skip_offsets):
        """
        [CF] 根据采样索引批量加载视频帧。
        
        Returns:
            sampled_list: PIL Image 列表
        """
        sampled_list = []
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step
        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            sampled_list = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
        except:
            raise RuntimeError('Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list, directory, duration))
        return sampled_list

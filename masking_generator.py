# [CF] 2026-04-10:
# 这个文件实现了 VideoMAE 的核心掩码策略 —— "管道掩码" (Tube Masking)。
# 它的作用是生成一个布尔掩码 (mask)，告诉模型哪些 patches 要被遮盖，哪些要保持可见。
# 与普通 MAE 对每帧独立掩码不同，VideoMAE 的管道掩码会沿着时间轴覆盖相同的空间位置，
# 从而强制模型学习物体的运动信息，防止从相邻帧"作弊"。

import numpy as np

class TubeMaskingGenerator:
    """
    [CF] 管道掩码生成器
    
    核心思想：生成一个空间掩码模式，然后将它"复制粘贴"到所有时间帧上。
    这样，被遮盖的 patch 在视频中形成了一条条沿着时间轴的"管道"。
    """
    def __init__(self, input_size, mask_ratio):
        """
        [CF] 初始化管道掩码生成器
        
        Args:
            input_size (tuple): 一个三元组 (T, H, W)，表示：
                - T: 时间维度的 patch 数量。例如，16帧视频，tubelet_size=2，则 T=8
                - H: 空间高度上的 patch 数量。例如 224x224 帧，patch_size=16，则 H=14
                - W: 空间宽度上的 patch 数量。例如 W=14
                这个 input_size 实际上就是 run_mae_pretraining.py 中计算好的 args.window_size
            mask_ratio (float): 掩码比例，通常在 0.75 到 0.9 之间。
                VideoMAE 推荐使用极高的掩码率（如 0.9）来加大任务难度。
        """
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        """
        [CF] 定义对象的字符串表示，方便打印调试信息。
        """
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        """
        [CF] 使得类的实例可以像函数一样被调用。
        每次调用都会生成一个全新的、随机的管道掩码。
        
        Returns:
            np.ndarray: 一个长度为 total_patches 的一维布尔数组。
                其中 1 (True) 表示该位置的 patch 被遮盖，
                0 (False) 表示该位置的 patch 保持可见。
        """
        # [CF] 1. 为"单帧"生成一个随机掩码模式
        # 首先创建一个数组，包含 (1 - mask_ratio) 比例的 0（可见）
        # 和 mask_ratio 比例的 1（被掩码）
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        # [CF] 随机打乱这个数组，为单帧生成一个随机的空间掩码分布
        np.random.shuffle(mask_per_frame)

        # [CF] 2. 将单帧掩码沿时间轴复制 T 次，形成"管道"
        # np.tile 将 mask_per_frame 在时间维度上重复 self.frames 次。
        # 这意味着对于所有时间帧，被掩码的 patch 在空间位置上是完全一致的。
        # 最后 flatten() 将其展平为一个一维数组。
        mask = np.tile(mask_per_frame, (self.frames,1)).flatten()
        return mask 

# [CF] ============================================================================
# [CF] 补充说明：为什么叫 "Tube" (管道)？
# [CF] 假设一个简单的例子：T=2 (两帧), H=2, W=2 (每帧4个patch)，mask_ratio=0.5。
# [CF] 
# [CF] 1. num_patches_per_frame = 4
# [CF] 2. num_masks_per_frame = 2
# [CF] 3. 单帧打乱后：mask_per_frame = [1, 0, 0, 1] (假设的结果)
# [CF] 4. 沿时间复制后：
# [CF]    - 第1帧：patch0=掩码, patch1=可见, patch2=可见, patch3=掩码
# [CF]    - 第2帧：patch0=掩码, patch1=可见, patch2=可见, patch3=掩码
# [CF] 
# [CF] 可以看到，相同的空间位置 (patch 0 和 patch 3) 在所有时间帧上都被掩码了。
# [CF] 这就好像在视频中切出了两条贯穿时间轴的"管道"，模型完全看不到管道内的任何信息。
# [CF] 
# [CF] 相比之下，如果使用 RandomMaskingGenerator (对每帧独立掩码)，
# [CF] 那么一个 patch 在第 t 帧被掩码，但在第 t+1 帧可能是可见的，
# [CF] 模型就可以轻易地从相邻帧"抄"到答案，导致信息泄露，无法有效学习运动特征。
# [CF] ============================================================================
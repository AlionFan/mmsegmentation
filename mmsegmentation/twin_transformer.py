# deeplabv3plus_swin_tiny_cityscapes_80k.py

_base_ = [
    # '../_base_/models/deeplabv3plus_r50-d8.py',  # 基础模型配置
    '../_base_/datasets/cityscapes.py',          # 数据集配置
    '../_base_/default_runtime.py',               # 运行时默认配置
    '../_base_/schedules/schedule_80k.py'        # 训练迭代数和学习率设置
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 1024)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size_divisor=32   # ✅保证图片尺寸能被32整除
)


# 修改模型结构为基于Swin Transformer Tiny
model = dict(
    type='EncoderDecoder',
    backbone = dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        patch_size=4,
        window_size=7,
        embed_dims=96,
        depths=[2, 2, 18, 2],       # 每个stage的Transformer层数
        num_heads=[3, 6, 12, 24],   # 每个stage的注意力头数
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
        ),
    ),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=768,    # 高层特征来自 stage3
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        c1_in_channels=192,   # 低层特征来自 stage1
        c1_channels=48,
        # c1_in_index=1         # 👈 关键：指定低层通道来自 stage1 而非 stage0
    ),

    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,  # stage2
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
    ),

    # decode_head 只用最后stage输出，auxiliary用倒数第二stage
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# 优化器，学习率等保持默认，或按需调整
optimizer = dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }
    )
)

optimizer_config = dict()

lr_config = dict(
    policy='poly',
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)

runner = dict(type='IterBasedRunner', max_iters=80000)

# 数据增强可保留基础配置
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
)

# 设置随机种子（可选）
seed = 0

# 默认runtime配置已包含工作目录、日志和checkpoints保存配置


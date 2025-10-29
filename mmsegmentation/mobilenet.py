_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/cityscapes.py', 
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

# 修改模型配置
model = dict(
    backbone=dict(
        _delete_=True,
        type='MobileNetV3',
        arch='large',
        out_indices=(0, 1, 2, 3),  # 修正输出索引
    ),
    decode_head=dict(
        in_channels=160,        # 高层特征的通道数
        in_index=3,             # 使用 out_indices 中的第4个输出
        c1_in_channels=160,      # 低层特征的通道数
    ),
    auxiliary_head=None,
    data_preprocessor=dict(
        size_divisor=32,
        size=None
    )
)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
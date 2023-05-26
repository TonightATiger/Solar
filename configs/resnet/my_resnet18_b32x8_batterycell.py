_base_ = 'resnet18_8xb32_in1k.py'

# _deprecation_ = dict(
#     expected='resnet18_8xb32_in1k.py',
#     reference='https://github.com/open-mmlab/mmclassification/pull/508',
# )
# model = dict(
#     type='ImageClassifier',
#     backbone=dict(
#         type='ResNet',
#         depth=18,
#         num_stages=4,
#         out_indices=(3, ),
#         style='pytorch'),
#     neck=dict(type='GlobalAveragePooling'),
#     head=dict(
#         type='LinearClsHead',
#         num_classes=6,
#         in_channels=512,
#         loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
#         topk=(1, 5),
#     ))
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=4,
              #loss=dict(type='LabelSmoothLoss',loss_weight=1.0,label_smooth_val=0.1,num_classes=17),
              loss=dict(type='CrossEntropyLoss',loss_weight=1.0)
              ),
)
dataset_type = 'BatteryCell'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='RandomResizedCrop', size=64, scale=(0.8, 1.2)),
#     dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
#     dict(
#         type='Normalize',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         to_rgb=True),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='ToTensor', keys=['gt_label']),
#     dict(type='Collect', keys=['img', 'gt_label'])
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', size=(64, -1)),
#     dict(
#         type='Normalize',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         to_rgb=True),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='Collect', keys=['img'])
# ]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type='BatteryCell',
        data_prefix=r'D:\LSTEST\openMMLab\DCP-AIXU\DATA\15-8-PLClassifier\data_split\train',
        ann_file=
        r'D:\LSTEST\openMMLab\DCP-AIXU\DATA\15-8-PLClassifier\meta\train.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', size=128, scale=(0.8, 1.2)),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='BatteryCell',
        data_prefix=r'D:\LSTEST\openMMLab\DCP-AIXU\DATA\15-8-PLClassifier\data_split\val',
        ann_file=
        r'D:\LSTEST\openMMLab\DCP-AIXU\DATA\15-8-PLClassifier\meta\val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(128, -1)),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='BatteryCell',
        data_prefix=r'D:\LSTEST\openMMLab\DCP-AIXU\DATA\15-8-PLClassifier\data_split\val',
        ann_file=
        r'D:\LSTEST\openMMLab\DCP-AIXU\DATA\15-8-PLClassifier\meta\val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(128, -1)),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
evaluation = dict(interval=1, metric='accuracy', save_best='auto')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-04)
optimizer_config = dict(grad_clip=None)
# lr_config = dict(policy='step', gamma=0.98, step=1)
lr_config = dict(policy='step', step=[30, 60, 90])
# lr_config = dict(
#     policy='CosineAnnealing',
#     min_lr=0,
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=1e-4)
runner = dict(type='EpochBasedRunner', max_epochs=300)
# runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(interval=1)
log_config = dict(interval=150, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
# log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1),('val', 1)]
work_dir = '../work-dir/resnet_0316'
gpu_ids = [0]
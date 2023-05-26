# dataset settings
dataset_type = 'BatteryCell'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=128, scale=(0.8, 1.2)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(128, -1)),
    # dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix=r'D:\LSTEST\openMMLab\DCP-AIXU\DATA\15-8-PLClassifier\data_split\train',
        ann_file=r'D:\LSTEST\openMMLab\DCP-AIXU\DATA\15-8-PLClassifier\meta\train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=r'D:\LSTEST\openMMLab\DCP-AIXU\DATA\15-8-PLClassifier\data_split\val',
        ann_file=r'D:\LSTEST\openMMLab\DCP-AIXU\DATA\15-8-PLClassifier\meta\val.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix=r'D:\LSTEST\openMMLab\DCP-AIXU\DATA\15-8-PLClassifier\data_split\val',
        ann_file=r'D:\LSTEST\openMMLab\DCP-AIXU\DATA\15-8-PLClassifier\meta\val.txt',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')

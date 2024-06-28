weight = 'exp/tgnet/semseg-pt-v3m1-0-tgnet-fps-good/model/model_best.pth'
resume = False
evaluate = True
test_only = False
seed = 28215351
save_path = 'exp/tgnet/semseg-pt-v3m1-0-tgnet-fps-good'
num_worker = 16
batch_size = 96
batch_size_val = None
batch_size_test = None
epoch = 1000
eval_epoch = 200
sync_bn = False
enable_amp = True
empty_cache = False
find_unused_parameters = False
mix_prob = 0
param_dicts = [dict(keyword='block', lr=0.0002)]
hooks = [
    dict(type='CheckpointLoader'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='TgnetEvaluatorFPS'),
    dict(type='CheckpointSaver', save_freq=None),
    dict(type='PreciseEvaluator', test_last=False)
]
train = dict(type='DefaultTrainer')
test = dict(type='SemSegTester', verbose=True)
model = dict(
    type='TgnetSegmentor',
    num_classes=17,
    backbone_out_channels=64,
    backbone=dict(
        type='PT-v3m1-tgnet',
        in_channels=6,
        order=['z', 'z-trans', 'hilbert', 'hilbert-trans'],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=('nuScenes', 'SemanticKITTI', 'Waymo')),
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1),
        dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=-1)
    ])
optimizer = dict(type='AdamW', lr=0.002, weight_decay=0.005)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.002, 0.0002],
    pct_start=0.04,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=100.0)
dataset_type = 'TgnetDataset'
data_root = 'data/tgnet_resize_dataset'
ignore_index = -1
names = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
    '14', '15', '16'
]
data = dict(
    num_classes=17,
    ignore_index=-1,
    names=[
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
        '13', '14', '15', '16'
    ],
    train=dict(
        type='TgnetDataset',
        split='train',
        data_root='data/tgnet_resize_dataset',
        transform=[
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'normal', 'segment'),
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'normal', 'id', 'jaw'),
                feat_keys=('coord', 'normal'))
        ],
        test_mode=False,
        ignore_index=-1,
        loop=5),
    val=dict(
        type='TgnetDataset',
        split='val',
        data_root='data/tgnet_resize_dataset',
        transform=[
            dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'normal', 'segment'),
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'normal', 'id', 'jaw'),
                feat_keys=('coord', 'normal'))
        ],
        test_mode=False,
        ignore_index=-1),
    test=dict(
        type='TgnetDataset',
        split='test',
        data_root='data/tgnet_resize_dataset',
        transform=[],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='test',
                return_grid_coord=True,
                keys=('coord', 'normal')),
            crop=None,
            post_transform=[
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'grid_coord', 'index', 'normal', 'id',
                          'jaw'),
                    feat_keys=('coord', 'normal'))
            ],
            aug_transform=[]),
        ignore_index=-1))

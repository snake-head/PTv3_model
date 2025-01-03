_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 96  # bs: total bs in all gpus
mix_prob = 0
empty_cache = False
enable_amp = True

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="TgnetEvaluatorFPS"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# model settings
model = dict(
    type="TgnetSegmentor",
    num_classes=17,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1-tgnet",
        in_channels=6,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
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
        pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)


# scheduler settings
epoch = 1000
eval_epoch = 200
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = [dict(keyword="block", lr=0.0002)]


# dataset settings
dataset_type = "TgnetDataset"
data_root = "data/tgnet_resize_dataset"
# data_root = "data/tgnet_dataset"


ignore_index = -1
names = [
    # 牙龈
    "0",
 
    # 
    "1",  
    "2",  
    "3",  
    "4",  
    "5",  
    "6",  
    "7",  
    "8",  
 
    # 
    "9",  
    "10",  
    "11",  
    "12",  
    "13",  
    "14",  
    "15",  
    "16",  


]
# names = [
#     # 牙龈
#     "0",
 
#     # 右上
#     "11",  
#     "12",  
#     "13",  
#     "14",  
#     "15",  
#     "16",  
#     "17",  
#     "18",  
 
#     # 左上
#     "21",  
#     "22",  
#     "23",  
#     "24",  
#     "25",  
#     "26",  
#     "27",  
#     "28",  

#     # 左下
#     "31",  
#     "32",  
#     "33", 
#     "34",  
#     "35",  
#     "36",  
#     "37",  
#     "38", 
    
#     # 右下
#     "41",  
#     "42",  
#     "43",  
#     "44",  
#     "45",  
#     "46",  
#     "47",  
#     "48", 
# ]

data = dict(
    num_classes=17,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="x", p=0.5),
            # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="y", p=0.5),
            # dict(type="PointClip", point_cloud_range=(-75.2, -75.2, -4, 75.2, 75.2, 2)),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            # dict(type="RandomFlip", p=0.5),
            # dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                keys=("coord", "normal", "segment"),
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode="random"),
            # dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "normal", "id", "jaw"),
                # keys=("coord", "segment", "offset_vector"),
                feat_keys=("coord", "normal"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            # dict(type="PointClip", point_cloud_range=(-75.2, -75.2, -4, 75.2, 75.2, 2)),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                keys=("coord", "normal", "segment"),
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "normal", "id", "jaw"),
                # keys=("coord", "segment", "offset_vector"),
                feat_keys=("coord", "normal"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            # dict(type="PointClip", point_cloud_range=(-75.2, -75.2, -4, 75.2, 75.2, 2)),
            # dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            
            # dict(
            #     type="GridSample",
            #     grid_size=0.025,
            #     hash_type="fnv",
            #     mode="test",
            #     keys=("coord", "normal", "segment"),
            #     # keys=("coord", "normal"),
            #     return_inverse=True,
            # ),
            # dict(type="ToTensor"),
            # dict(
            #     type="Collect",
            #     keys=("coord", "grid_coord", "segment"),
            #     feat_keys=("coord", "normal"),
            # ),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                keys=("coord", "normal"),
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index","normal", "id", "jaw"),
                    feat_keys=("coord", "normal"),
                ),
            ],
            aug_transform=[
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[0],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     )
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[1 / 2],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     )
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[1],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     )
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[3 / 2],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     )
                # ],
            ],
        ),
        ignore_index=ignore_index,
    ),
)
DistributedDataParallel(
  (module): DefaultSegmentorV2(
    (seg_head): Linear(in_features=64, out_features=17, bias=True)
    (backbone): PointTransformerV3(
      (embedding): Embedding(
        (stem): PointSequential(
          (conv): SubMConv3d(6, 32, kernel_size=[5, 5, 5], stride=[1, 1, 1], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.Native)
          (norm): BatchNorm1d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (act): GELU(approximate='none')
        )
      )
      (enc): PointSequential(
        (enc0): PointSequential(
          (block0): Block(
            (cpe): PointSequential(
              (0): SubMConv3d(32, 32, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
              (1): Linear(in_features=32, out_features=32, bias=True)
              (2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
            )
            (norm1): PointSequential(
              (0): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
            )
            (attn): SerializedAttention(
              (qkv): Linear(in_features=32, out_features=96, bias=True)
              (proj): Linear(in_features=32, out_features=32, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (norm2): PointSequential(
              (0): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
            )
            (mlp): PointSequential(
              (0): MLP(
                (fc1): Linear(in_features=32, out_features=128, bias=True)
                (act): GELU(approximate='none')
                (fc2): Linear(in_features=128, out_features=32, bias=True)
                (drop): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): PointSequential(
              (0): Identity()
            )
          )
          (block1): Block(
            (cpe): PointSequential(
              (0): SubMConv3d(32, 32, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
              (1): Linear(in_features=32, out_features=32, bias=True)
              (2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
            )
            (norm1): PointSequential(
              (0): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
            )
            (attn): SerializedAttention(
              (qkv): Linear(in_features=32, out_features=96, bias=True)
              (proj): Linear(in_features=32, out_features=32, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (norm2): PointSequential(
              (0): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
            )
            (mlp): PointSequential(
              (0): MLP(
                (fc1): Linear(in_features=32, out_features=128, bias=True)
                (act): GELU(approximate='none')
                (fc2): Linear(in_features=128, out_features=32, bias=True)
                (drop): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): PointSequential(
              (0): DropPath(drop_prob=0.023)
            )
          )
        )
        (enc1): PointSequential(
          (down): SerializedPooling(
            (proj): Linear(in_features=32, out_features=64, bias=True)
            (norm): PointSequential(
              (0): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
            (act): PointSequential(
              (0): GELU(approximate='none')
            )
          )
          (block0): Block(
            (cpe): PointSequential(
              (0): SubMConv3d(64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
              (1): Linear(in_features=64, out_features=64, bias=True)
              (2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            )
            (norm1): PointSequential(
              (0): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            )
            (attn): SerializedAttention(
              (qkv): Linear(in_features=64, out_features=192, bias=True)
              (proj): Linear(in_features=64, out_features=64, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (norm2): PointSequential(
              (0): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            )
            (mlp): PointSequential(
              (0): MLP(
                (fc1): Linear(in_features=64, out_features=256, bias=True)
                (act): GELU(approximate='none')
                (fc2): Linear(in_features=256, out_features=64, bias=True)
                (drop): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): PointSequential(
              (0): DropPath(drop_prob=0.046)
            )
          )
          (block1): Block(
            (cpe): PointSequential(
              (0): SubMConv3d(64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
              (1): Linear(in_features=64, out_features=64, bias=True)
              (2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            )
            (norm1): PointSequential(
              (0): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            )
            (attn): SerializedAttention(
              (qkv): Linear(in_features=64, out_features=192, bias=True)
              (proj): Linear(in_features=64, out_features=64, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (norm2): PointSequential(
              (0): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            )
            (mlp): PointSequential(
              (0): MLP(
                (fc1): Linear(in_features=64, out_features=256, bias=True)
                (act): GELU(approximate='none')
                (fc2): Linear(in_features=256, out_features=64, bias=True)
                (drop): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): PointSequential(
              (0): DropPath(drop_prob=0.069)
            )
          )
        )
        (enc2): PointSequential(
          (down): SerializedPooling(
            (proj): Linear(in_features=64, out_features=128, bias=True)
            (norm): PointSequential(
              (0): BatchNorm1d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
            (act): PointSequential(
              (0): GELU(approximate='none')
            )
          )
          (block0): Block(
            (cpe): PointSequential(
              (0): SubMConv3d(128, 128, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
              (1): Linear(in_features=128, out_features=128, bias=True)
              (2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            )
            (norm1): PointSequential(
              (0): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            )
            (attn): SerializedAttention(
              (qkv): Linear(in_features=128, out_features=384, bias=True)
              (proj): Linear(in_features=128, out_features=128, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (norm2): PointSequential(
              (0): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            )
            (mlp): PointSequential(
              (0): MLP(
                (fc1): Linear(in_features=128, out_features=512, bias=True)
                (act): GELU(approximate='none')
                (fc2): Linear(in_features=512, out_features=128, bias=True)
                (drop): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): PointSequential(
              (0): DropPath(drop_prob=0.092)
            )
          )
          (block1): Block(
            (cpe): PointSequential(
              (0): SubMConv3d(128, 128, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
              (1): Linear(in_features=128, out_features=128, bias=True)
              (2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            )
            (norm1): PointSequential(
              (0): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            )
            (attn): SerializedAttention(
              (qkv): Linear(in_features=128, out_features=384, bias=True)
              (proj): Linear(in_features=128, out_features=128, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (norm2): PointSequential(
              (0): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            )
            (mlp): PointSequential(
              (0): MLP(
                (fc1): Linear(in_features=128, out_features=512, bias=True)
                (act): GELU(approximate='none')
                (fc2): Linear(in_features=512, out_features=128, bias=True)
                (drop): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): PointSequential(
              (0): DropPath(drop_prob=0.115)
            )
          )
        )
        (enc3): PointSequential(
          (down): SerializedPooling(
            (proj): Linear(in_features=128, out_features=256, bias=True)
            (norm): PointSequential(
              (0): BatchNorm1d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
            (act): PointSequential(
              (0): GELU(approximate='none')
            )
          )
          (block0): Block(
            (cpe): PointSequential(
              (0): SubMConv3d(256, 256, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
              (1): Linear(in_features=256, out_features=256, bias=True)
              (2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
            (norm1): PointSequential(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
            (attn): SerializedAttention(
              (qkv): Linear(in_features=256, out_features=768, bias=True)
              (proj): Linear(in_features=256, out_features=256, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (norm2): PointSequential(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
            (mlp): PointSequential(
              (0): MLP(
                (fc1): Linear(in_features=256, out_features=1024, bias=True)
                (act): GELU(approximate='none')
                (fc2): Linear(in_features=1024, out_features=256, bias=True)
                (drop): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): PointSequential(
              (0): DropPath(drop_prob=0.138)
            )
          )
          (block1): Block(
            (cpe): PointSequential(
              (0): SubMConv3d(256, 256, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
              (1): Linear(in_features=256, out_features=256, bias=True)
              (2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
            (norm1): PointSequential(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
            (attn): SerializedAttention(
              (qkv): Linear(in_features=256, out_features=768, bias=True)
              (proj): Linear(in_features=256, out_features=256, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (norm2): PointSequential(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
            (mlp): PointSequential(
              (0): MLP(
                (fc1): Linear(in_features=256, out_features=1024, bias=True)
                (act): GELU(approximate='none')
                (fc2): Linear(in_features=1024, out_features=256, bias=True)
                (drop): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): PointSequential(
              (0): DropPath(drop_prob=0.162)
            )
          )
          (block2): Block(
            (cpe): PointSequential(
              (0): SubMConv3d(256, 256, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
              (1): Linear(in_features=256, out_features=256, bias=True)
              (2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
            (norm1): PointSequential(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
            (attn): SerializedAttention(
              (qkv): Linear(in_features=256, out_features=768, bias=True)
              (proj): Linear(in_features=256, out_features=256, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (norm2): PointSequential(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
            (mlp): PointSequential(
              (0): MLP(
                (fc1): Linear(in_features=256, out_features=1024, bias=True)
                (act): GELU(approximate='none')
                (fc2): Linear(in_features=1024, out_features=256, bias=True)
                (drop): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): PointSequential(
              (0): DropPath(drop_prob=0.185)
            )
          )
          (block3): Block(
            (cpe): PointSequential(
              (0): SubMConv3d(256, 256, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
              (1): Linear(in_features=256, out_features=256, bias=True)
              (2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
            (norm1): PointSequential(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
            (attn): SerializedAttention(
              (qkv): Linear(in_features=256, out_features=768, bias=True)
              (proj): Linear(in_features=256, out_features=256, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (norm2): PointSequential(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
            (mlp): PointSequential(
              (0): MLP(
                (fc1): Linear(in_features=256, out_features=1024, bias=True)
                (act): GELU(approximate='none')
                (fc2): Linear(in_features=1024, out_features=256, bias=True)
                (drop): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): PointSequential(
              (0): DropPath(drop_prob=0.208)
            )
          )
          (block4): Block(
            (cpe): PointSequential(
              (0): SubMConv3d(256, 256, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
              (1): Linear(in_features=256, out_features=256, bias=True)
              (2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
            (norm1): PointSequential(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
            (attn): SerializedAttention(
              (qkv): Linear(in_features=256, out_features=768, bias=True)
              (proj): Linear(in_features=256, out_features=256, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (norm2): PointSequential(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
            (mlp): PointSequential(
              (0): MLP(
                (fc1): Linear(in_features=256, out_features=1024, bias=True)
                (act): GELU(approximate='none')
                (fc2): Linear(in_features=1024, out_features=256, bias=True)
                (drop): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): PointSequential(
              (0): DropPath(drop_prob=0.231)
            )
          )
          (block5): Block(
            (cpe): PointSequential(
              (0): SubMConv3d(256, 256, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
              (1): Linear(in_features=256, out_features=256, bias=True)
              (2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
            (norm1): PointSequential(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
            (attn): SerializedAttention(
              (qkv): Linear(in_features=256, out_features=768, bias=True)
              (proj): Linear(in_features=256, out_features=256, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (norm2): PointSequential(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
            (mlp): PointSequential(
              (0): MLP(
                (fc1): Linear(in_features=256, out_features=1024, bias=True)
                (act): GELU(approximate='none')
                (fc2): Linear(in_features=1024, out_features=256, bias=True)
                (drop): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): PointSequential(
              (0): DropPath(drop_prob=0.254)
            )
          )
        )
        (enc4): PointSequential(
          (down): SerializedPooling(
            (proj): Linear(in_features=256, out_features=512, bias=True)
            (norm): PointSequential(
              (0): BatchNorm1d(512, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
            (act): PointSequential(
              (0): GELU(approximate='none')
            )
          )
          (block0): Block(
            (cpe): PointSequential(
              (0): SubMConv3d(512, 512, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
              (1): Linear(in_features=512, out_features=512, bias=True)
              (2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            )
            (norm1): PointSequential(
              (0): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            )
            (attn): SerializedAttention(
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (norm2): PointSequential(
              (0): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            )
            (mlp): PointSequential(
              (0): MLP(
                (fc1): Linear(in_features=512, out_features=2048, bias=True)
                (act): GELU(approximate='none')
                (fc2): Linear(in_features=2048, out_features=512, bias=True)
                (drop): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): PointSequential(
              (0): DropPath(drop_prob=0.277)
            )
          )
          (block1): Block(
            (cpe): PointSequential(
              (0): SubMConv3d(512, 512, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
              (1): Linear(in_features=512, out_features=512, bias=True)
              (2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            )
            (norm1): PointSequential(
              (0): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            )
            (attn): SerializedAttention(
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (norm2): PointSequential(
              (0): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            )
            (mlp): PointSequential(
              (0): MLP(
                (fc1): Linear(in_features=512, out_features=2048, bias=True)
                (act): GELU(approximate='none')
                (fc2): Linear(in_features=2048, out_features=512, bias=True)
                (drop): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): PointSequential(
              (0): DropPath(drop_prob=0.300)
            )
          )
        )
      )
      (dec): PointSequential(
        (dec3): PointSequential(
          (up): SerializedUnpooling(
            (proj): PointSequential(
              (0): Linear(in_features=512, out_features=256, bias=True)
              (1): BatchNorm1d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): GELU(approximate='none')
            )
            (proj_skip): PointSequential(
              (0): Linear(in_features=256, out_features=256, bias=True)
              (1): BatchNorm1d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): GELU(approximate='none')
            )
          )
          (block0): Block(
            (cpe): PointSequential(
              (0): SubMConv3d(256, 256, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
              (1): Linear(in_features=256, out_features=256, bias=True)
              (2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
            (norm1): PointSequential(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
            (attn): SerializedAttention(
              (qkv): Linear(in_features=256, out_features=768, bias=True)
              (proj): Linear(in_features=256, out_features=256, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (norm2): PointSequential(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
            (mlp): PointSequential(
              (0): MLP(
                (fc1): Linear(in_features=256, out_features=1024, bias=True)
                (act): GELU(approximate='none')
                (fc2): Linear(in_features=1024, out_features=256, bias=True)
                (drop): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): PointSequential(
              (0): DropPath(drop_prob=0.300)
            )
          )
          (block1): Block(
            (cpe): PointSequential(
              (0): SubMConv3d(256, 256, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
              (1): Linear(in_features=256, out_features=256, bias=True)
              (2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
            (norm1): PointSequential(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
            (attn): SerializedAttention(
              (qkv): Linear(in_features=256, out_features=768, bias=True)
              (proj): Linear(in_features=256, out_features=256, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (norm2): PointSequential(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
            (mlp): PointSequential(
              (0): MLP(
                (fc1): Linear(in_features=256, out_features=1024, bias=True)
                (act): GELU(approximate='none')
                (fc2): Linear(in_features=1024, out_features=256, bias=True)
                (drop): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): PointSequential(
              (0): DropPath(drop_prob=0.257)
            )
          )
        )
        (dec2): PointSequential(
          (up): SerializedUnpooling(
            (proj): PointSequential(
              (0): Linear(in_features=256, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): GELU(approximate='none')
            )
            (proj_skip): PointSequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): GELU(approximate='none')
            )
          )
          (block0): Block(
            (cpe): PointSequential(
              (0): SubMConv3d(128, 128, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
              (1): Linear(in_features=128, out_features=128, bias=True)
              (2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            )
            (norm1): PointSequential(
              (0): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            )
            (attn): SerializedAttention(
              (qkv): Linear(in_features=128, out_features=384, bias=True)
              (proj): Linear(in_features=128, out_features=128, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (norm2): PointSequential(
              (0): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            )
            (mlp): PointSequential(
              (0): MLP(
                (fc1): Linear(in_features=128, out_features=512, bias=True)
                (act): GELU(approximate='none')
                (fc2): Linear(in_features=512, out_features=128, bias=True)
                (drop): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): PointSequential(
              (0): DropPath(drop_prob=0.214)
            )
          )
          (block1): Block(
            (cpe): PointSequential(
              (0): SubMConv3d(128, 128, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
              (1): Linear(in_features=128, out_features=128, bias=True)
              (2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            )
            (norm1): PointSequential(
              (0): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            )
            (attn): SerializedAttention(
              (qkv): Linear(in_features=128, out_features=384, bias=True)
              (proj): Linear(in_features=128, out_features=128, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (norm2): PointSequential(
              (0): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            )
            (mlp): PointSequential(
              (0): MLP(
                (fc1): Linear(in_features=128, out_features=512, bias=True)
                (act): GELU(approximate='none')
                (fc2): Linear(in_features=512, out_features=128, bias=True)
                (drop): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): PointSequential(
              (0): DropPath(drop_prob=0.171)
            )
          )
        )
        (dec1): PointSequential(
          (up): SerializedUnpooling(
            (proj): PointSequential(
              (0): Linear(in_features=128, out_features=64, bias=True)
              (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): GELU(approximate='none')
            )
            (proj_skip): PointSequential(
              (0): Linear(in_features=64, out_features=64, bias=True)
              (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): GELU(approximate='none')
            )
          )
          (block0): Block(
            (cpe): PointSequential(
              (0): SubMConv3d(64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
              (1): Linear(in_features=64, out_features=64, bias=True)
              (2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            )
            (norm1): PointSequential(
              (0): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            )
            (attn): SerializedAttention(
              (qkv): Linear(in_features=64, out_features=192, bias=True)
              (proj): Linear(in_features=64, out_features=64, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (norm2): PointSequential(
              (0): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            )
            (mlp): PointSequential(
              (0): MLP(
                (fc1): Linear(in_features=64, out_features=256, bias=True)
                (act): GELU(approximate='none')
                (fc2): Linear(in_features=256, out_features=64, bias=True)
                (drop): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): PointSequential(
              (0): DropPath(drop_prob=0.129)
            )
          )
          (block1): Block(
            (cpe): PointSequential(
              (0): SubMConv3d(64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
              (1): Linear(in_features=64, out_features=64, bias=True)
              (2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            )
            (norm1): PointSequential(
              (0): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            )
            (attn): SerializedAttention(
              (qkv): Linear(in_features=64, out_features=192, bias=True)
              (proj): Linear(in_features=64, out_features=64, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (norm2): PointSequential(
              (0): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            )
            (mlp): PointSequential(
              (0): MLP(
                (fc1): Linear(in_features=64, out_features=256, bias=True)
                (act): GELU(approximate='none')
                (fc2): Linear(in_features=256, out_features=64, bias=True)
                (drop): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): PointSequential(
              (0): DropPath(drop_prob=0.086)
            )
          )
        )
        (dec0): PointSequential(
          (up): SerializedUnpooling(
            (proj): PointSequential(
              (0): Linear(in_features=64, out_features=64, bias=True)
              (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): GELU(approximate='none')
            )
            (proj_skip): PointSequential(
              (0): Linear(in_features=32, out_features=64, bias=True)
              (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): GELU(approximate='none')
            )
          )
          (block0): Block(
            (cpe): PointSequential(
              (0): SubMConv3d(64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
              (1): Linear(in_features=64, out_features=64, bias=True)
              (2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            )
            (norm1): PointSequential(
              (0): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            )
            (attn): SerializedAttention(
              (qkv): Linear(in_features=64, out_features=192, bias=True)
              (proj): Linear(in_features=64, out_features=64, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (norm2): PointSequential(
              (0): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            )
            (mlp): PointSequential(
              (0): MLP(
                (fc1): Linear(in_features=64, out_features=256, bias=True)
                (act): GELU(approximate='none')
                (fc2): Linear(in_features=256, out_features=64, bias=True)
                (drop): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): PointSequential(
              (0): DropPath(drop_prob=0.043)
            )
          )
          (block1): Block(
            (cpe): PointSequential(
              (0): SubMConv3d(64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
              (1): Linear(in_features=64, out_features=64, bias=True)
              (2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            )
            (norm1): PointSequential(
              (0): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            )
            (attn): SerializedAttention(
              (qkv): Linear(in_features=64, out_features=192, bias=True)
              (proj): Linear(in_features=64, out_features=64, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (norm2): PointSequential(
              (0): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            )
            (mlp): PointSequential(
              (0): MLP(
                (fc1): Linear(in_features=64, out_features=256, bias=True)
                (act): GELU(approximate='none')
                (fc2): Linear(in_features=256, out_features=64, bias=True)
                (drop): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): PointSequential(
              (0): Identity()
            )
          )
        )
      )
    )
  )
)
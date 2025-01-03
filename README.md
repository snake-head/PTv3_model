<!--
 * @Description: 
 * @Version: 1.0
 * @Autor: ZhuYichen
 * @Date: 2024-12-30 14:30:11
 * @LastEditors: ZhuYichen
 * @LastEditTime: 2025-01-03 12:56:17
-->
## 目前采用模型
模型名：**semseg-pt-v3m1-0-tgnet-fps-good**

代码路径：\\172.16.200.7\Data.MIVA\ZhuYichen\PTv3_model

测试数据路径：\\172.16.200.7\Data.MIVA\ZhuYichen\PTv3_model\data\request_modeldata

模型路径：\\172.16.200.7\Data.MIVA\ZhuYichen\semseg-pt-v3m1-0-tgnet-fps-good\model\model_best.pth

## 环境配置
### 依赖安装
```sh
# 创建conda环境
git clone --recursive https://github.com/Pointcept/PointTransformerV3.git
conda create -n pointcept python=3.8 -y
conda activate pointcept
conda install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
# We use CUDA 11.8 and PyTorch 2.1.0 for our development of PTv3
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

cd libs/pointops
python setup.py install
cd ../..

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu118  # choose version match your local cuda version

# Open3D (visualization, optional)
pip install open3d

pip install flash-attn
```

## 目前后端调用模型命令
```sh
sh scripts/infer.sh -p python -g 1 -d tgnet -n semseg-pt-v3m1-0-tgnet-fps-good -w model_best -i 'data/request_modeldata/{id}/' -o 'data/request_result/' -k true
```
## 推理 infer
```sh
sh scripts/infer.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -n ${EXP_NAME} -w ${CHECKPOINT_NAME} -i ${INPUT_PATH} -o ${OUTPUT_PATH} -k ${IF_USE_KNN}
```
如:
```sh
sh scripts/infer.sh -p python -g 1 -d tgnet -n semseg-pt-v3m1-0-tgnet-fps-full-test -w model_best -i 'data/tgnet_fulldataset_whole_norm' -o 'data/result_of_test' -k true
```

## 测试 test

```sh
sh scripts/test.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -n ${EXP_NAME} -w ${CHECKPOINT_NAME}
```
如:
```sh
sh scripts/test.sh -p python -g 1 -d tgnet -n semseg-pt-v3m1-0-tgnet-fps-full-test -w model_best
```

## 恢复训练 resume train

```sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# Script (Recommended)
# simply add "-r true"
sh scripts/train.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -c ${CONFIG_NAME} -n ${EXP_NAME} -r true
# Direct
export PYTHONPATH=./
python tools/train.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH} resume=True weight=${CHECKPOINT_PATH}
```

## 训练 train
```sh
sh scripts/train.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -c ${CONFIG_NAME} -n ${EXP_NAME}
```
如:
```sh
sh scripts/train.sh -p python -g 4 -d tgnet -c semseg-pt-v3m1-0-tgnet-fps-mask1 -n semseg-pt-v3m1-0-tgnet-fps-mask1
```

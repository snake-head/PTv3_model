## 目前采用模型
模型名：**semseg-pt-v3m1-0-tgnet-fps-good**

服务器路径：/mnt/data.coronaryct.1/YeYangfan/Src/Pointcept/exp/tgnet/semseg-pt-v3m1-0-tgnet-fps-good

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

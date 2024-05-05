# 训练
```sh
sh scripts/train.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -c ${CONFIG_NAME} -n ${EXP_NAME}
```
```sh
sh scripts/train.sh -p python -d scannet -c semseg-pt-v2m2-0-base -n semseg-pt-v2m2-0-base
```
```sh
sh scripts/train.sh -p python -g 4 -d tgnet -c semseg-pt-v3m1-0-base -n semseg-pt-v3m1-0-base
```

```sh
sh scripts/train.sh -p python -g 1 -d tgnet -c semseg-pt-v3m1-0-base -n semseg-pt-v3m1-0-base
```
## 训练tgnet

```sh
sh scripts/train.sh -p python -g 4 -d tgnet -c semseg-pt-v3m1-0-tgnet-fps -n semseg-pt-v3m1-0-tgnet-fps
```
```sh
sh scripts/train.sh -p python -g 4 -d tgnet -c semseg-pt-v3m1-0-tgnet-fps -n semseg-pt-v3m1-0-tgnet-fps -r true
```

## 训练tgnet-test
    ```sh
    sh scripts/train.sh -p python -g 1 -d tgnet -c semseg-pt-v3m1-0-tgnet-fps-test -n semseg-pt-v3m1-0-tgnet-fps-test
    ```

## 训练tgnet-simple

```sh
sh scripts/train.sh -p python -g 4 -d tgnet -c semseg-pt-v3m1-0-tgnet-fps-simple -n semseg-pt-v3m1-0-tgnet-fps-simple
```


## 训练tgnet-full
完成fps的训练
```sh
sh scripts/train.sh -p python -g 4 -d tgnet -c semseg-pt-v3m1-0-tgnet-fps-full -n semseg-pt-v3m1-0-tgnet-fps-full
```

## 训练tgnet-full-test
在这里优化模型的代码，区分train val test的pipeline并完善评估

```sh
sh scripts/train.sh -p python -g 4 -d tgnet -c semseg-pt-v3m1-0-tgnet-fps-full-test -n semseg-pt-v3m1-0-tgnet-fps-full-test
```
## 训练tgnet-full-tester
调试tester

```sh
sh scripts/train.sh -p python -g 4 -d tgnet -c semseg-pt-v3m1-0-tgnet-fps-full-tester -n semseg-pt-v3m1-0-tgnet-fps-full-tester
```

## 恢复训练

```sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# Script (Recommended)
# simply add "-r true"
sh scripts/train.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -c ${CONFIG_NAME} -n ${EXP_NAME} -r true
# Direct
export PYTHONPATH=./
python tools/train.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH} resume=True weight=${CHECKPOINT_PATH}
```

## 测试

```sh
sh scripts/test.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -n ${EXP_NAME} -w ${CHECKPOINT_NAME}
    
sh scripts/test.sh -p python -g 1 -d tgnet -n semseg-pt-v3m1-0-base -w model_best

sh scripts/test.sh -p python -g 1 -d tgnet -n semseg-pt-v3m1-0-tgnet-fps-full-test -w model_best

sh scripts/test.sh -p python -g 1 -d tgnet -n semseg-pt-v3m1-0-tgnet-fps-full-tester -w model_best

```



# 设计网络

- 先把网络print出来看一下结构，也便于后续连接mlp输出offset和mask
  看论文
- 预处理里面把点的offset做
- 在trainer中自定义一个方法，替代原本的build mode
- 定义一个类 在类中用原本build的方法 build出两个模型，并作为该类的第一第二模块，在该类的forward中定义两个模块的用法，
      - 首先要想办法把模块的mlp连出来
      - 在这个方法中，build这个类



## 网络的思路

### fps模型

####     第一模块
   - head

        - mask or label 得到seg1

        - offset

####     根据mask 和offset得到偏移后的点

   - offset 的mlp
        - tag test 
        	- > 64，32，16，8，3 
        	  >
        	  > trainloss 0.1， valloss 0.2-10，有点过拟合
        	
        - tag simple
        
          - > 64 16 3  
            >
            > grid_size = 0.02 clustering = DBSCAN(eps=2, min_samples=150).fit(moved_point)
            >
            > Train result: loss_seg: 0.2854 loss_offset: 1.7448 loss_dir: 0.0112 
            >
            > Val result: mIoU/mAcc/allAcc/offset/dir 0.7772/0.8229/0.9605/36.0602/0.6780.
        
          - > grid_size = 0.05 clustering = DBSCAN(eps=0.8, min_samples=10).fit(moved_point)
        
        
        
   - 聚类DBSCAN

        - 得到质心和每个牙齿
        - 将聚类的分割结果覆盖在seg_logits上，此时得到seg2
          - 注意logits是未softmax的，改的数值差别要尽可能大，如0和10
        - 对于每个牙齿的单位空间，输入第二个模块
        - 为了后续的处理，这里制作的单牙point需要包含多个属性
          - coord (bn,3)
          - grid_coord (bn,3)
          - segment (bn,)
          - offset (b,)
          - feat (bn,64)
          - chosed_mask (bn,)
          - label (bn,)
          - mask_target (bn,)
          - mask_seg_logits (bn,)


#### 第二模块
- 这个backbone和前面的不一样，因为输入feat是64维
- head
  - mask
  - 这里的mask指该单位空间中，属于该牙齿的点，而不是该空间中，属于牙齿的点，以此优化聚类的结果
  - 对于该mask中True的点，赋予他们同一label，False赋予label 0
    - 注意logits是未sigmoid的，要输出大于0认定true
  - 此时得到seg3，也是最终结果
    - 注意logits是未softmax的，改的数值差别要尽可能大，如0和10






    聚类DBSCAN
        预处理后的牙齿坐标在[-1,1]之间
    得到质心和每个牙齿
    对于每个牙齿的单位空间，输入第二个模块
    输出
        label
    （根据jaw将输出的label 还原到fdi编码的label

### bdl模型

​    使用已有的fps模型得到label
​    采样其中位于边界的部分

    第一次输出，
        mask or label
        offset
    根据mask 和offset得到偏移后的点
    聚类
    得到质心和每个牙齿
    对于每个牙齿的单位空间，输入第二个模块
    输出
        label
    （根据jaw将输出的label 还原到fdi编码的label

结合fps和bdl的结果，对每个点进行邻域投票，即为最终结果


替换网络，要model和serialization


结果
tag：fulldata
    纯ptv3 200代 
    Val result: mIoU/mAcc/allAcc 0.7990/0.8503/0.9478
    感觉还没到收敛
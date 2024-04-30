训练
sh scripts/train.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -c ${CONFIG_NAME} -n ${EXP_NAME}

sh scripts/train.sh -p python -d scannet -c semseg-pt-v2m2-0-base -n semseg-pt-v2m2-0-base

sh scripts/train.sh -p python -g 4 -d tgnet -c semseg-pt-v3m1-0-base -n semseg-pt-v3m1-0-base


sh scripts/train.sh -p python -g 1 -d tgnet -c semseg-pt-v3m1-0-base -n semseg-pt-v3m1-0-base


测试


sh scripts/test.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -n ${EXP_NAME} -w ${CHECKPOINT_NAME}



sh scripts/test.sh -p python -g 1 -d tgnet -n semseg-pt-v3m1-0-base -w model_best




设计网络
先把网络print出来看一下结构，也便于后续连接mlp输出offset和mask
看论文
预处理里面把点的offset做好
在trainer中自定义一个方法，替代原本的build model
定义一个类 在类中用原本build的方法 build出两个模型，并作为该类的第一第二模块，在该类的forward中定义两个模块的用法，
    首先要想办法把模块的mlp连出来
在这个方法中，build这个类



网络的思路
fps模型
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

bdl模型
    使用已有的fps模型得到label
    采样其中位于边界的部分

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
"""
Default Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from .builder import DATASETS, build_dataset
from .transform import Compose, TRANSFORMS
import pointops


@DATASETS.register_module()
class TgnetDataset(Dataset):
    def __init__(
        self,
        split="train",
        data_root="data/tgnet_dataset",
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        ignore_index=-1,
        infer_mode=False,
    ):
        print('hhhhhhhhhhh')
        super(TgnetDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.infer_mode = infer_mode
        self.test_cfg = test_cfg if test_mode else None

        if test_mode:
            self.test_voxelize = (
                TRANSFORMS.build(self.test_cfg.voxelize)
                if self.test_cfg.voxelize is not None
                else None
            )
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop)
                if self.test_cfg.crop is not None
                else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            # self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]
            self.aug_transform = None
            
        self.data_list = self.get_data_list()
        print('eeeee',self.data_list)
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )
        print('dataset init done')

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*.pth"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.pth"))
        else:
            raise NotImplementedError
        print(data_list)
        return data_list

    def get_data(self, idx):
        data = torch.load(self.data_list[idx % len(self.data_list)])
        fileName = self.data_list[idx % len(self.data_list)].split('/')[-1]
        # print('fileName',fileName)
        if not 'id' in data.keys():
            id = fileName.split('_')[0]
        else:
            id = data['id']
        if not 'jaw' in data.keys():
            jaw = fileName.split('.')[0].split('_')[-1]
        else:
            jaw = data['jaw']
        coord = data["coord"]
        # color = data["color"]
        normal = data["normal"]
        # todo 去掉offset vector
        # print('dataloader', coord.shape)
        if "labels" in data.keys() and self.infer_mode==False:
            segment = data["labels"].reshape([-1])
            
            # offset_vector = data['offset']
            # print('get offset vector',offset_vector.shape)
        else:
            segment = np.ones(coord.shape[0]) * -1
        # 最远点采样
        # if True:
            # idx = pointops.farthest_point_sampling()
        data_dict = dict(coord=coord, normal=normal, segment=segment, id=id, jaw=jaw)
        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split(".")[0]

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        
        # print('train',type(data_dict))
        # print('here offset:',data_dict.keys())
        # for key in data_dict:
        #     print(key,data_dict[key].shape,data_dict[key])
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        # print(type(data_dict))
        # print('herehere:',data_dict.pop("segment"))
        # print(idx)
        # print('name',self.get_data_name(idx))
        result_dict = dict(
            segment=data_dict.pop("segment"), name=self.get_data_name(idx)
        )
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        data_dict_list = []
        if self.aug_transform is not None and self.aug_transform != []:
            for aug in self.aug_transform:
                data_dict_list.append(aug(deepcopy(data_dict)))
        else:
            data_dict_list.append(deepcopy(data_dict))

        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                fragment_list += data_part

        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])
        result_dict["fragment_list"] = fragment_list
        return result_dict
    
    
    def prepare_infer_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        # print('transform后',data_dict['coord'].shape)
        # print(type(data_dict))
        # print('herehere:',data_dict.pop("segment"))
        # print(idx)
        # print('name',self.get_data_name(idx))
        result_dict = dict(
            name=self.get_data_name(idx),
            coord=data_dict['coord'],
            id = data_dict['id'],
            jaw = data_dict['jaw']
        )
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        data_dict_list = []

        data_dict_list.append(deepcopy(data_dict))

        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                fragment_list += data_part
                
        # print('voxelize后',data_dict['coord'].shape, data_dict['grid_coord'].shape)
        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])
        result_dict["fragment_list"] = fragment_list
        print('return前',result_dict.keys())
        return result_dict
    
    
    def __getitem__(self, idx):
        if self.infer_mode:
            return self.prepare_infer_data(idx)
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop


# @DATASETS.register_module()
# class ConcatDataset(Dataset):
#     def __init__(self, datasets, loop=1):
#         super(ConcatDataset, self).__init__()
#         self.datasets = [build_dataset(dataset) for dataset in datasets]
#         self.loop = loop
#         self.data_list = self.get_data_list()
#         logger = get_root_logger()
#         logger.info(
#             "Totally {} x {} samples in the concat set.".format(
#                 len(self.data_list), self.loop
#             )
#         )

#     def get_data_list(self):
#         data_list = []
#         for i in range(len(self.datasets)):
#             data_list.extend(
#                 zip(
#                     np.ones(len(self.datasets[i])) * i, np.arange(len(self.datasets[i]))
#                 )
#             )
#         return data_list

#     def get_data(self, idx):
#         dataset_idx, data_idx = self.data_list[idx % len(self.data_list)]
#         return self.datasets[dataset_idx][data_idx]

#     def get_data_name(self, idx):
#         dataset_idx, data_idx = self.data_list[idx % len(self.data_list)]
#         return self.datasets[dataset_idx].get_data_name(data_idx)

#     def __getitem__(self, idx):
#         return self.get_data(idx)

#     def __len__(self):
#         return len(self.data_list) * self.loop

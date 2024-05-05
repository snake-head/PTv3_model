import torch.nn as nn

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model

from sklearn.cluster import DBSCAN  
import torch
import numpy as np
from collections import Counter
import torch.nn.functional as F 

@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        seg_logits = self.seg_head(point.feat)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        feat = self.backbone(input_dict)
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)


# 以下为自定义
@MODELS.register_module()
class TgnetSegmentor(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        logger=None,
    ):
        super().__init__()
        self.logger = logger
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.offset_head = nn.Sequential(
            # nn.Linear(backbone_out_channels, 3)
            nn.Linear(backbone_out_channels, 16),
            # nn.Linear(32,16),
            nn.Linear(16,3),
            # nn.Linear(8,3),
            # if num_classes > 0
            # else nn.Identity()
        )
        self.mask_head = nn.Sequential(
            nn.Linear(backbone_out_channels, 1),
            # nn.Sigmoid() 
        )
        self.first_module = build_model(backbone)
        # 第二模块的输入feat维度是64
        backbone.in_channels = 64
        self.second_module = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.criteria_offset = nn.MSELoss(size_average=True)
        self.criteria_dir = nn.CosineSimilarity(dim=1, eps=1e-6) 
        self.criteria_mask = F.binary_cross_entropy_with_logits
        self.logger.info(f"tgnetsegmentor build完毕")
        # print('tgnetsegmentor build完毕')

    def forward(self, input_dict):
        print('forward')
        
        # 预处理输入
        point = Point(input_dict)
        # 输入第一模块
        point = self.first_module(point)
        
        
        # 得到第一模块的seg、offset结果
        seg_logits = self.seg_head(point.feat)
        # self.logger.info(f'seg_logits{seg_logits}')
        offset_logits = self.offset_head(point.feat)
        
        # 根据第一模块的结果，进行点偏移和聚类，准备第二模块的输入
        # 偏移并聚类
        # done 通过loss查看cls seg的结果，这个不做优化也无法优化
        
        # clustering, cls_to_label, cls_seg_logits, cls_segment_logits = self.get_cluster(point,seg_logits=seg_logits,offset_logits=offset_logits, input_dict=input_dict)
        
        
        # 输入第二模块
        # 循环每个聚类，也就是每个batch的牙齿要输入十几个单牙
        # 拆分一个batch里的单牙，作为一个point输入后，将得到的数据根据offset拼接回一个batch牙
        # result_of_second = self.split_tooth_in_one_batch(point, clustering, cls_to_label, offset_logits, cls_seg_logits, cls_segment_logits)
        # if result_of_second is not None:
        #     loss_mask, mask_logits, mask_segment_logits = result_of_second
        # else:
        #     loss_mask, mask_logits, mask_segment_logits = None, None, None
            
            
        # 制作单牙
        single_tooth_point = None
        if self.training:
            # 训练时不用聚类，但是可以查看数据
            # single_tooth_point = self.get_single_tooth_point(point, phase='val', seg_logits=cls_segment_logits)
            single_tooth_point = self.get_single_tooth_point(point, phase='train')     
        else:
            # todo 后续改为val
            # 先根据seg聚类
            cls_segment_logits = self.get_cluster_val(point=point, seg_logits=seg_logits, offset_logits=offset_logits)
            single_tooth_point = self.get_single_tooth_point(point, phase='val', seg_logits=cls_segment_logits)
            # clustering, cls_to_label, cls_seg_logits, cls_segment_logits = self.get_cluster(point,seg_logits=seg_logits,offset_logits=offset_logits, input_dict=input_dict)
            
        # self.logger.info(f'{point}')
        # self.logger.info(f'{offset_logits}')
        # 保存一个测试数据 
        # if point['grid_coord'].get_device() == 0:
        #         data = {}
        #         data['jaw'] = point['jaw']
        #         data['id'] = point['id']
        #         all_data = {}
        #         all_data['istrain'] = self.training
        #         all_data['grid_coord'] = point['grid_coord']
        #         all_data['offset'] = offset_logits
        #         all_data['segment'] = point['segment']
        #         all_data['normal'] = point['normal']
        # 输入第二模块
        if single_tooth_point is not None and 'segment' in input_dict.keys():
            single_tooth_point_output = self.second_module(single_tooth_point)
            mask_logits = self.mask_head(single_tooth_point_output.feat)
            # 计算loss_mask
            loss_mask = self.get_loss_mask(single_tooth_point_output, mask_logits=mask_logits)
            # 注意，目前在seg1上更新
            mask_segment_logits = self.get_mask_segment_logits(single_tooth_point_output, seg_logits=seg_logits, mask_logits=mask_logits)
            
            # 保存一个测试数据 
            # if point['grid_coord'].get_device() == 0:
            #     data = {}
            #     data['jaw'] = point['jaw']
            #     data['id'] = point['id']
            #     all_data = {}
            #     all_data['istrain'] = self.training
            #     all_data['grid_coord'] = point['grid_coord']
            #     all_data['offset'] = offset_logits
            #     all_data['segment'] = point['segment']
            #     all_data['normal'] = point['normal']
            #     if not self.training:
            #         all_data['cls_segment_logits'] = cls_segment_logits
            #     all_data['batch_offset'] = point['offset']
            #     all_data['coord'] = point['coord']
            #     all_data['batch'] = point['batch']
            #     all_data['seg_logits'] = seg_logits
            #     all_data['mask_segment_logits'] = mask_segment_logits
                
            #     single_data = {}
            #     single_data['single_grid_coord'] = single_tooth_point_output['grid_coord']
            #     single_data['single_batch'] = single_tooth_point_output['batch']
                
            #     single_data['mask_logits'] = mask_logits
            #     single_data['mask_target'] = single_tooth_point_output['mask_target']
            #     single_data['single_label'] = single_tooth_point_output['label']
            #     single_data['single_mask_all_to_crop'] = single_tooth_point_output['mask_all_to_crop']
            #     single_data['single_segment'] = single_tooth_point_output['segment']
            #     data['all'] = all_data 
            #     data['single'] = single_data
                
            #     torch.save(data,'full_crop_data.pth')
        else:
            single_tooth_point_output = self.second_module(single_tooth_point)
            mask_logits = self.mask_head(single_tooth_point_output.feat)
            # 计算loss_mask
            # loss_mask = self.get_loss_mask(single_tooth_point_output, mask_logits=mask_logits)
            # 注意，目前在seg1上更新
            mask_segment_logits = self.get_mask_segment_logits(single_tooth_point_output, seg_logits=seg_logits, mask_logits=mask_logits)

        
        
            
            

        
        # train
        if self.training:
            
            # 计算loss seg
            self.logger.info(f'正常 type {seg_logits.dtype} {input_dict["segment"].dtype}')
            loss_seg = self.criteria(seg_logits, input_dict["segment"])
            self.logger.info(f'train loss_seg,{loss_seg}')
            
            # 计算offset target
            offset_target = self.get_offset_target(point)
            # 计算loss offset
            loss_offset = self.get_loss_offset(offset_logits=offset_logits, offset_target=offset_target)
            self.logger.info(f'train loss_offset,{loss_offset}')
            
            # 计算dir loss
            if True:
                offset_pred_norm = torch.norm(offset_logits,dim=1).view(-1,1)
                offset_pred_dir = torch.div(offset_logits, offset_pred_norm)
                mask_pred = offset_pred_norm > 0.01
                offset_target_norm = torch.norm(offset_target, dim=1).view(-1,1)
                offset_target_dir = torch.div(offset_target, offset_target_norm)
                mask_target = offset_target_norm > 0.01
                mask = mask_pred & mask_target
                mask = mask.squeeze()
                dir_mat = torch.sum(offset_pred_dir * offset_target_dir, dim=1)
                dir_mat = dir_mat - 1
                dir_mat = dir_mat * dir_mat
                dir_mat = dir_mat[mask]
                loss_dir = torch.div(torch.sum(dir_mat),dir_mat.shape[0])
                # self.logger.info(f"手动计算loss dir{loss_dir}, 用类计算的{self.get_loss_dir(offset_logits=offset_logits, offset_target=offset_target).mean()}")
                # loss_dir = self.get_loss_dir(offset_logits=offset_logits, offset_target=offset_target).mean()
            self.logger.info(f'train loss_dir, {loss_dir}')
            
            
            # 计算seg2的loss
            # self.logger.info(f'type {cls_segment_logits.dtype} {input_dict["segment"].dtype}{cls_segment_logits.shape} {input_dict["segment"].shape}')
            # loss_seg_cls = self.criteria(cls_segment_logits, input_dict["segment"])
            # self.logger.info(f'train loss_seg_cls {loss_seg_cls}')
            # self.logger.info(f'seg2比seg1进步 {loss_seg- loss_seg_cls}')
            
            
            

                
            # 计算seg3的loss
            if mask_segment_logits is not None:
                loss_seg_mask = self.criteria(mask_segment_logits , input_dict["segment"])
                self.logger.info(f'train loss_seg_mask {loss_seg_mask}')
                self.logger.info(f'  seg3比seg1进步 {loss_seg-loss_seg_mask}')
            if loss_mask is not None:
                self.logger.info(f'train loss_mask {loss_mask}')
                return dict(loss_seg=loss_seg, loss_offset=loss_offset, loss_dir=loss_dir, loss_mask=loss_mask)
            else:
                return dict(loss_seg=loss_seg, loss_offset=loss_offset, loss_dir=loss_dir)
        # eval
        elif "segment" in input_dict.keys():
            
            # 计算loss seg
            loss_seg = self.criteria(seg_logits, input_dict["segment"])
            self.logger.info(f'val loss_seg,{loss_seg}')
            
            # 计算offset target
            offset_target = self.get_offset_target(point)
            # 计算loss offset
            loss_offset = self.get_loss_offset(offset_logits=offset_logits, offset_target=offset_target)
            self.logger.info(f'val loss_offset,{loss_offset}')
            
            # 计算dir loss
            if True:
                offset_pred_norm = torch.norm(offset_logits,dim=1).view(-1,1)
                offset_pred_dir = torch.div(offset_logits, offset_pred_norm)
                mask_pred = offset_pred_norm > 0.01
                offset_target_norm = torch.norm(offset_target, dim=1).view(-1,1)
                offset_target_dir = torch.div(offset_target, offset_target_norm)
                mask_target = offset_target_norm > 0.01
                mask = mask_pred & mask_target
                mask = mask.squeeze()
                dir_mat = torch.sum(offset_pred_dir * offset_target_dir, dim=1)
                dir_mat = dir_mat - 1
                dir_mat = dir_mat * dir_mat
                dir_mat = dir_mat[mask]
                loss_dir = torch.div(torch.sum(dir_mat),dir_mat.shape[0])
                # self.logger.info(f"手动计算loss dir{loss_dir}, 用类计算的{self.get_loss_dir(offset_logits=offset_logits, offset_target=offset_target).mean()}")
                # loss_dir = self.get_loss_dir(offset_logits=offset_logits, offset_target=offset_target).mean()
            self.logger.info(f'train loss_dir, {loss_dir}')
            
            
            # 计算seg2
            loss_seg_cls = self.criteria(cls_segment_logits, input_dict["segment"])
            self.logger.info(f'train loss_seg_cls {loss_seg_cls}')
            self.logger.info(f'  seg2比seg1进步 {loss_seg- loss_seg_cls}')
            if mask_segment_logits is not None:
                loss_seg_mask = self.criteria(mask_segment_logits , input_dict["segment"])
                self.logger.info(f'train loss_seg_mask {loss_seg_mask}')
                self.logger.info(f'  seg3比seg2进步 {loss_seg_cls-loss_seg_mask}')
                self.logger.info(f'train loss_mask {loss_mask}')
            return dict(loss_seg=loss_seg, loss_offset=loss_offset, loss_dir=loss_dir, loss_mask=loss_mask, loss_seg_cls=loss_seg_cls, loss_seg_mask=loss_seg_mask, seg_logits=seg_logits, offset_logits=offset_logits)
        # test
        else:
            return dict(seg_logits=seg_logits, offset_logits=offset_logits,cls_segment_logits=cls_segment_logits, mask_segment_logits=mask_segment_logits)
        
    def split_tooth_in_one_batch(self, point, clustering, cls_to_label, offset_logits, cls_seg_logits, cls_segment_logits):
        self.logger.info(f"开始制作第二模块的输入")
        
        batch_offset = point['offset']
        last_offset = 0
        mask_feat = torch.zeros(point['feat'].shape)
        # 三级分割label以二级为基础
        mask_seg_logits = cls_segment_logits
        
        # 制作所有batch内所有单牙的point，相当于每个batch拆成多个batch，并修正字典
        single_tooth_point = {}
        
        # 对于每一个batch牙
        # todo 优化遍历方案，可以用batch mask
        batch_mask = point['batch']
        batch_sum = batch_mask[-1] + 1
        for b in range(batch_sum):
            # self.logger.info(f"拆分batch {b}")
            if not bool(cls_to_label[b].keys()):
                self.logger.info(f"暂无聚类 退出单牙 batch {b}")
                continue
            # 取出该batch
            batch_mask_b = (batch_mask == b)
            # full_mask = torch.clone(batch_mask_b)
            # 'coord', 'grid_coord', 'segment', 'offset_vector', 'offset', 'feat
            coor_b = point['coord'][batch_mask_b]
            grid_coord_b = point['grid_coord'][batch_mask_b].to(torch.float)
            segment_b = point['segment'][batch_mask_b]
            # batch_offset_b = point['offset'][last_offset:batch_offset[b]].cpu().detach().numpy()
            feat_b = point['feat'][batch_mask_b]
            print('here',coor_b.dtype, grid_coord_b.dtype,segment_b.dtype)
            
            offset_logits_b = offset_logits[batch_mask_b]
            
            cls_to_label_b = cls_to_label[b]
            self.logger.info(f"batch {b} 遍历聚类 聚类有{cls_to_label_b.keys()}")
            # 根据聚类取出每个单牙
            for cls in cls_to_label_b.keys():
                # self.logger.info(f"取出该聚类 {cls}")
                label = cls_to_label_b[cls]
                tooth_label = torch.clone(segment_b)
                # 对于该batch的label，根据聚类结果更新其label，也就是seg2
                tooth_label[segment_b!=0][cls_seg_logits[b] == label] = label
                grid_coord_thislabel = grid_coord_b[tooth_label==label]
                center = torch.mean(grid_coord_thislabel,dim=0)
                print(center)
                offset = grid_coord_thislabel - center
                norms = torch.norm(offset, dim=1) 
                max_magnitude_vector = offset[torch.argmax(norms)]
                print(max_magnitude_vector,torch.norm(max_magnitude_vector))
                
                # 计算每个点与目标点的距离  
                distances = torch.norm(grid_coord_b - center, dim=1)  
                # 选出距离小于 某值 的点  
                chosed_mask = (distances < torch.norm(max_magnitude_vector)*1.1)
                chosed_points_grid_coord = grid_coord_b[chosed_mask] 
                chosed_points_coord = coor_b[chosed_mask]
                # 移动到中心
                # todo 第二个gridcoord是移动到中心还是移动到左下角为原点？
                # todo grid_coord 移动了，那么feat还能用吗
                # todo 要归一化嘛
                # todo coord要移动的话，要移动到coord的center
                # todo gridcoord 必变为long，如何避免都挤在一起？ 不归一化而是归十化？
                chosed_points_grid_coord = chosed_points_grid_coord - center
                chosed_points_coord = (chosed_points_coord - center) * 10
                chosed_points_grid_coord = chosed_points_grid_coord.to(torch.long)
                print(chosed_points_grid_coord.dtype)
                
                # 制作该牙的label
                label_type = torch.full((segment_b[chosed_mask].shape), label).cuda()
                # self.logger.info(f" {single_tooth_point.keys()} batch {b}, cls {cls}")
                
                # 制作point
                # 使用 torch.cat 函数连接这两个张量 
                # full_mask[full_mask==True] &= chosed_mask
                if not 'coord' in single_tooth_point.keys():
                    self.logger.info(f"制作单牙 ")
                    single_tooth_point.update({
                        'coord': chosed_points_coord,
                        'grid_coord': chosed_points_grid_coord,
                        'segment': segment_b[chosed_mask],
                        'offset': torch.tensor([chosed_points_grid_coord.shape[0]]).cuda(),
                        'feat': feat_b[chosed_mask],
                        'chosed_mask': chosed_mask,
                        'label_type': label_type,
                        'mask_target': (segment_b[chosed_mask] == label),
                        'original_batch': torch.tensor([b]).cuda(),
                        
                    })
                else:
                    point_amount = single_tooth_point['offset'][-1] if len(single_tooth_point['offset']) > 1 else single_tooth_point['offset'] 
                    single_tooth_point.update({
                        'coord': torch.cat((single_tooth_point['coord'], chosed_points_coord), dim=0),
                        'grid_coord': torch.cat((single_tooth_point['grid_coord'], chosed_points_grid_coord), dim=0),
                        'segment': torch.cat((single_tooth_point['segment'], segment_b[chosed_mask]), dim=0),
                        'offset': torch.cat((single_tooth_point['offset'], torch.tensor([point_amount + chosed_points_grid_coord.shape[0]]).cuda()), dim=0),
                        'feat': torch.cat((single_tooth_point['feat'], feat_b[chosed_mask]), dim=0),
                        'chosed_mask': torch.cat((single_tooth_point['chosed_mask'], chosed_mask), dim=0),
                        'label_type': torch.cat((single_tooth_point['label_type'], label_type), dim=0),
                        'mask_target': torch.cat((single_tooth_point['mask_target'], (segment_b[chosed_mask] == label)), dim=0),
                        'original_batch': torch.cat((single_tooth_point['original_batch'],torch.tensor([b]).cuda()), dim=0),
                    })

        
            last_offset = batch_offset[b]
        
        self.logger.info(f"single tooth keys{single_tooth_point.keys()}")
        # self.logger.info(f"single tooth point{single_tooth_point.feat[0]},{single_tooth_point.coord[0],{single_tooth_point.grid_coord[0]},single_tooth_point}")
        
        # 如果没有分出单牙,冻结模块
        if not bool(single_tooth_point.keys()) or len(single_tooth_point['offset'])==1:
            self.logger.info(f"暂无单牙")
            for name, param in self.mask_head.named_parameters():
                # print(name, param)
                param.requires_grad = False
                # print(name, param)
            
            return
        else:
            self.logger.info(f"有单牙")
            for name, param in self.mask_head.named_parameters():
                # print(name, param)
                param.requires_grad = True
                # print(name, param)
        self.logger.info(f"single tooth point{single_tooth_point['feat'][0]},{single_tooth_point['coord'][0]},{single_tooth_point['grid_coord'][0]},{single_tooth_point}")
        
        # 新point制作完毕 
        single_tooth_point = Point(single_tooth_point)
        self.logger.info(f"single tooth point keys{single_tooth_point.keys()}")
        self.logger.info(f"single tooth point Point{single_tooth_point.feat[0]},{single_tooth_point.coord[0]},{single_tooth_point.grid_coord[0]},{single_tooth_point}")
        
        # 输入第二模块
        single_tooth_point_output = self.second_module(single_tooth_point)
        self.logger.info(f'第二模块输出,{single_tooth_point_output}')
        # mask_feat[single_tooth_point_output['batch_to_single_mask']] = single_tooth_point_output.feat
        
        # 计算loss mask 这个要优化
        mask_logits = self.mask_head(single_tooth_point_output.feat)
        mask_target = single_tooth_point.mask_target
        mask_target = mask_target.to(torch.float)
        self.logger.info(f'计算loss mask,{mask_logits.dtype},{mask_target.dtype}')
        loss_mask = self.criteria_mask(mask_logits, mask_target.unsqueeze(1) )
        
        # 对于mask_seg_logits,需要重新计算
        mask_segment_logits = self.get_mask_seg_logits(point, single_tooth_point_output, cls_segment_logits, mask_logits)
        # tooth_batch_offset = single_tooth_point_output.offset
        # last_tooth_batch = 0
        # for tooth_batch in range(len(tooth_batch_offset)):
        #     original_batch_b = single_tooth_point_output.original_batch[tooth_batch]
        # trust_mask = mask_logits > 0.5
        # mask_seg_logits[trust_mask] = single_tooth_point_output.label[trust_mask]
            
            
            
        return loss_mask, mask_logits, mask_segment_logits

    # 计算seg3
    def get_mask_seg_logits(self, point, single_tooth_point_output, cls_segment_logits, mask_logits):
        tooth_batch_offset = single_tooth_point_output.offset
        tooth_batch_mask = single_tooth_point_output.batch
        last_tooth_batch = 0
        last_full_index = 0
        mask_segment_logits = torch.clone(cls_segment_logits)
        for tooth_batch in range(len(tooth_batch_offset)):
            original_batch_b = single_tooth_point_output.original_batch[tooth_batch]
            mask_from_all_to_full = (point.batch == original_batch_b)
            mask_from_full_to_one = single_tooth_point_output.chosed_mask[last_full_index:last_full_index+torch.sum(mask_from_all_to_full).item()]
            last_full_index = last_full_index+torch.sum(mask_from_all_to_full).item()
            mask_logits_one = (mask_logits[tooth_batch_mask == tooth_batch] > 0.5).squeeze()
            # self.logger.info(f'device信息{mask_segment_logits.dtype}{mask_from_all_to_full.dtype}{mask_from_full_to_one.dtype}{mask_logits_one.dtype}')
            # self.logger.info(f'maskdneg信息{mask_segment_logits.shape}{mask_from_all_to_full.shape}{mask_from_full_to_one.shape}{mask_logits_one.shape}')
            # self.logger.info(f'{single_tooth_point_output.label_type.shape},{single_tooth_point_output.label_type.dtype}')
            # self.logger.info(f'{single_tooth_point_output.label_type.shape},{single_tooth_point_output.label_type.dtype}')
            # self.logger.info(f'maskfulltoone{mask_from_full_to_one.masked_scatter_(mask_from_full_to_one, mask_logits_one).shape}')
            # self.logger.info(f'maskalltofall{mask_from_all_to_full.masked_scatter_(mask_from_all_to_full, mask_from_full_to_one.masked_scatter_(mask_from_full_to_one, mask_logits_one)).shape}')
            # self.logger.info(f'{single_tooth_point_output.label_type.shape}')
            # self.logger.info(f'{(tooth_batch_mask == tooth_batch).shape}')
            # self.logger.info(f'{single_tooth_point_output.label_type[tooth_batch_mask == tooth_batch].shape}')
            # self.logger.info(f'{(mask_logits_one).shape}')
            # self.logger.info(f'右侧{single_tooth_point_output.label_type[tooth_batch_mask == tooth_batch][mask_logits_one]}')
            
            # self.logger.info(f'here1{mask_segment_logits[mask_from_all_to_full.masked_scatter_(mask_from_all_to_full, mask_from_full_to_one.masked_scatter_(mask_from_full_to_one, mask_logits_one))].shape}')
            # self.logger.info(f'here2{single_tooth_point_output.label_type[tooth_batch_mask == tooth_batch][mask_logits_one].shape}')
            label = single_tooth_point_output.label_type[tooth_batch_mask == tooth_batch][0]
            refined_class = torch.zeros(17).to(mask_segment_logits.dtype).cuda()
            refined_class[label] = 1
            mask_segment_logits[mask_from_all_to_full.masked_scatter_(mask_from_all_to_full, mask_from_full_to_one.masked_scatter_(mask_from_full_to_one, mask_logits_one))] = refined_class
            last_tooth_batch = tooth_batch_offset[tooth_batch]
        return mask_segment_logits
    
    # 新的计算seg3
    def get_mask_segment_logits(self, single_tooth_point_output, seg_logits, mask_logits):
        # 注意！这里的seg_logits，在不同phase下采用不同的logits来检查性能
        # train用seg1
        # val  用seg2
        # test 用seg2
        tooth_batch_offset = single_tooth_point_output.offset
        tooth_batch_mask = single_tooth_point_output.batch
        label_type = single_tooth_point_output.label
        mask_all_to_crop = single_tooth_point_output.mask_all_to_crop
        
        mask_segment_logits = torch.clone(seg_logits)
        # 对于每一个cropped单牙
        for tooth_batch in range(len(tooth_batch_offset)):
            # 取出该牙认为的label
            label = label_type[tooth_batch]
            refined_class = torch.zeros(17).to(seg_logits.dtype).cuda()
            # print('refinedclass',refined_class, label)
            refined_class[label] = 10
            
            # 取出该牙的mask logits
            mask_logits_cropped = (mask_logits[tooth_batch_mask == tooth_batch] > 0).squeeze()
            
            # 更新mask_segment_logits
            # print(mask_all_to_crop[tooth_batch].shape)
            mask_all_to_update = mask_all_to_crop[tooth_batch].masked_scatter_(mask_all_to_crop[tooth_batch], mask_logits_cropped)
            mask_segment_logits[mask_all_to_update] = refined_class
        return mask_segment_logits
        
    
    def get_single_tooth_point(self, point, phase, seg_logits=None):
        # todo feat是否要改
        if phase == 'train' or phase == 'val':
            segment = point.segment
            if seg_logits is not None:
                segment = torch.argmax(seg_logits, dim=1) 
            batch_mask = point.batch
            coord = point.coord
            grid_coord = point.grid_coord
            feat = point.feat
            grid_coord = grid_coord.to(torch.float)
            single_tooth_point = {}
            # 对于每一个full
            for b in range(batch_mask[-1] + 1):
                # 找到每一个one
                for label in range(1, 17):
                    # 取出每一颗牙
                    mask_all_to_one = (batch_mask == b) & (segment == label)
                    # 如果无该牙齿
                    if not torch.any(mask_all_to_one):
                        continue
                    center = torch.mean(grid_coord[mask_all_to_one], dim=0)
                    offset_vector = grid_coord[mask_all_to_one] - center
                    # print('offset_vector',offset_vector)
                    # 找到模长最大的向量所在的索引
                    norms = torch.norm(offset_vector, dim=1)
                    # print('norms',norms)
                    max_norm_index = torch.argmax(norms)
                    crop_distance = norms[max_norm_index] 
                    
                    distance = torch.norm(grid_coord - center, dim=1)
                    mask_all_to_crop = (distance < crop_distance * 1.1) & (batch_mask == b)
                    
                    # 取出单位牙齿
                    grid_coord_cropped = grid_coord[mask_all_to_crop]
                    grid_coord_cropped -= center
                    grid_coord_cropped *= 10
                    grid_coord_cropped = grid_coord_cropped.to(torch.long)
                    
                    # 存入point
                    if not 'coord' in single_tooth_point:
                        single_tooth_point.update({
                            'coord': coord[mask_all_to_crop],
                            'grid_coord': grid_coord_cropped,
                            'segment': segment[mask_all_to_crop],
                            'offset': torch.tensor([grid_coord_cropped.shape[0]]).cuda(),
                            'feat': feat[mask_all_to_crop],
                            'mask_all_to_crop': mask_all_to_crop.unsqueeze(0),
                            'label': torch.tensor([label]),
                            'mask_target': (segment[mask_all_to_crop] == label),
                        })
                    else:
                        # 长度不定，用dim=0
                        point_amount = single_tooth_point['offset'][-1] if len(single_tooth_point['offset']) > 1 else single_tooth_point['offset'] 
                        single_tooth_point.update({
                            'coord': torch.cat((single_tooth_point['coord'], coord[mask_all_to_crop]), dim=0),
                            'grid_coord': torch.cat((single_tooth_point['grid_coord'], grid_coord_cropped), dim=0),
                            'segment': torch.cat((single_tooth_point['segment'],segment[mask_all_to_crop]), dim=0),
                            'offset': torch.cat((single_tooth_point['offset'], torch.tensor([point_amount + grid_coord_cropped.shape[0]]).cuda()), dim=0),
                            'feat': torch.cat((single_tooth_point['feat'], feat[mask_all_to_crop]), dim=0),
                            'mask_all_to_crop': torch.cat((single_tooth_point['mask_all_to_crop'], mask_all_to_crop.unsqueeze(0)), dim=0),
                            'label': torch.cat((single_tooth_point['label'], torch.tensor([label])), dim=0),
                            'mask_target': torch.cat((single_tooth_point['mask_target'], (segment[mask_all_to_crop] == label)), dim=0), 
                        })
            
            if not bool(single_tooth_point.keys()): 
                self.logger.info(f'暂无单牙')
                return None
            else: 
                single_tooth_point = Point(single_tooth_point)
                self.logger.info(f'单牙制作完成')
                return single_tooth_point
        # if phase == 'val':
            
        return None

        
    def get_loss_mask(self, single_tooth_point_output, mask_logits):
        self.logger.info(f"计算loss mask")
        mask_target = single_tooth_point_output.mask_target
        mask_target = mask_target.to(torch.float)
        self.logger.info(f'mask的logits,target数据类型,{mask_logits.dtype},{mask_target.dtype}')
        loss_mask = self.criteria_mask(mask_logits, mask_target.unsqueeze(1) )
        return loss_mask
        
            
    def get_loss_dir(self, offset_logits, offset_target):
        loss_dir = self.criteria_dir(offset_logits, offset_target)
        return loss_dir
        
        
    def get_loss_offset(self, offset_logits, offset_target):
        loss_offset = self.criteria_offset(offset_logits, offset_target)
        return loss_offset
    
    
    def get_offset_target(self, point):
        grid_coord = point['grid_coord']
        # print(grid_coord.dtype)
        grid_coord = grid_coord.to(torch.float)
        # print(grid_coord.dtype)
        segment = point['segment']
        offset_target = torch.zeros(grid_coord.shape).cuda()
        batch_mask = point['batch']
        # 对于每一个full
        for b in range(batch_mask[-1] + 1):
            # 找到每一个one
            for label in range(1, 17):
                mask_all_to_one = (batch_mask == b) & (segment == label)
                center = torch.mean(grid_coord[mask_all_to_one], dim=0)
                offset_target[mask_all_to_one] = center - grid_coord[mask_all_to_one]
        return offset_target
            

    
    # 只参考推理的聚类结果
    def get_cluster_val(self, point, seg_logits, offset_logits):
        self.logger.info('聚类开始')
        batch_offset = point['offset']
        batch_mask = point['batch']
        num = 0

        # cls_to_label = {}
        # cls_seg_logits = {}
        # cls_segment_logits = torch.clone(input_dict['segment']).to(torch.float)
        
        # 生成seg2，用于更新
        cls_segment_logits = torch.clone(seg_logits)
        
        # 如果输入是logits，转换成分类结果
        if seg_logits.dim() == 2 and seg_logits.shape[1] == 17:
            segment = torch.argmax(seg_logits, dim=1) 
        # 将认为是牙齿的点偏移
        grid_coord = point['grid_coord']
        moved_coord = grid_coord + offset_logits
        moved_coord = moved_coord
        # 取出每一个full，进行聚类
        for b in range(batch_offset.shape[0]):
            moved_coord_b = moved_coord[batch_mask == b].cpu().detach().numpy()
            segment_b = segment[batch_mask == b].cpu().detach().numpy()
            # 计算聚类
            clustering = DBSCAN(eps=2, min_samples=100).fit(moved_coord_b)
            self.logger.info(f'clust on ,{b},聚类数,{len(set(clustering.labels_)) - 1},应有,{len(set(segment_b))},点数,{np.count_nonzero(clustering.labels_ != -1)},应有,{np.count_nonzero(segment_b)}')
            # 对于每个聚类
            for cls in range(len(set(clustering.labels_)) - 1):
                cls_mask = (clustering.labels_ == cls)
                counter = Counter(segment_b[cls_mask])  
                most_common_element = counter.most_common(1)[0] 
                
                # 找到对应label
                most ,count = most_common_element
                label = most

                # 更新seg
                refined_class = torch.zeros(17).to(cls_segment_logits.dtype).cuda()
                refined_class[label] = 10
                mask_all_to_cls = (batch_mask == b).masked_scatter_((batch_mask == b), torch.from_numpy(cls_mask).cuda())
                # print(b,cls,label,batch_mask.shape)
                # print('here',batch_mask == b,cls_mask.shape,mask_all_to_cls)
                cls_segment_logits[mask_all_to_cls] = refined_class
                
        self.logger.info('聚类结束')  
        return cls_segment_logits
            
            
        
    
    
    def get_cluster(self, point, seg_logits, offset_logits, input_dict):
        self.logger.info('聚类开始')
        batch_offset = point['offset']
        batch_mask = point['batch']
        last_offset = 0
        num = 0

        cls_to_label = {}
        cls_seg_logits = {}
        # cls_segment_logits = torch.clone(input_dict['segment']).to(torch.float)
        
        cls_segment_logits = torch.clone(seg_logits)
        for b in range(batch_offset.shape[0]):
            # 取出该batch
            # todo 可以不用np的都转成tensor
            grid_coord_b = point['grid_coord'][last_offset:batch_offset[b]].cpu().detach().numpy()
            offset_logits_b = offset_logits[last_offset:batch_offset[b]].cpu().detach().numpy()
            segment_b = point['segment'][last_offset:batch_offset[b]].cpu().detach().numpy()
            mask = (segment_b != 0)
            moved_point = grid_coord_b + offset_logits_b
            moved_point = moved_point
            moved_point = moved_point[mask]
            
            # 计算聚类
            clustering = DBSCAN(eps=2, min_samples=100).fit(moved_point)
            self.logger.info(f'clust on ,{b},聚类数,{len(set(clustering.labels_))},应有,{len(set(segment_b))},点数,{np.count_nonzero(clustering.labels_ != -1)},应有,{np.count_nonzero(segment_b != 0)}')
            
            # 计算聚类正确率
            # todo 不同聚类指向同一个label，代码里会有覆盖的问题，可能性不大但是以后要解决
            label_to_cls = np.zeros(17) - 1
            cls_to_label_b = {}
            cls_seg_logits_b = np.zeros(segment_b[mask].shape)
            # 对于每个聚类，找到他们认为对应的标签
            for cls in range(len(set(clustering.labels_))-1):
                cls_mask = (clustering.labels_ == cls)
                counter = Counter(segment_b[mask][cls_mask])  
                most_common_element = counter.most_common(1)[0] 
                
                most ,count = most_common_element
                # print(most,count/np.count_nonzero(clustering.labels_ == cls))
                label = most
                label_to_cls[label] = cls
                cls_to_label_b[cls] = label
                cls_seg_logits_b[np.where(clustering.labels_ == cls)] = label
                
                refined_class = torch.zeros(17).to(cls_segment_logits.dtype).cuda()
                refined_class[label] = 1
                cls_segment_logits[last_offset:batch_offset[b]][mask][np.where(clustering.labels_ == cls)] = refined_class
            cls_to_label[b] = cls_to_label_b
            cls_seg_logits[b] = torch.from_numpy(cls_seg_logits_b).to(torch.long).cuda()
            # 对于每个label，计算它们的正确率
            for label in np.unique(segment_b):
                print('类型',label, '应有',np.count_nonzero(segment_b == label), '实有',np.count_nonzero(clustering.labels_ == label_to_cls[label]),'其中正确',np.sum((segment_b[mask]==label)&(cls_seg_logits_b == label)))
            print('聚类正确比例',(segment_b[mask] == cls_seg_logits_b).sum()/ len(cls_seg_logits_b))
            if len(set(clustering.labels_)) == len(set(segment_b)):
                print('聚类数正确')
                num += 1
            else:
                print('错误')
            
            last_offset = batch_offset[b]
        print('聚类数量正确率',num/batch_offset.shape[0])
        
        self.logger.info(f"聚类结束 ")
        # 这个返回目前只为测试数据返回
        return clustering, cls_to_label, cls_seg_logits, cls_segment_logits
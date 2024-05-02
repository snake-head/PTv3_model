import torch.nn as nn

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model

from sklearn.cluster import DBSCAN  
import torch
import numpy as np
from collections import Counter

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
    ):
        super().__init__()
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
        self.first_module = build_model(backbone)
        # self.second_module = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.criteria_offset = nn.MSELoss(size_average=True)
        self.criteria_dir = nn.CosineSimilarity(dim=1, eps=1e-6) 
        print('tgnetsegmentor build完毕')

    def forward(self, input_dict):
        print('forward')
        
        # 预处理输入
        point = Point(input_dict)
        # 输入第一模块
        point = self.first_module(point)
        
        
        # 得到第一模块的seg、offset结果
        seg_logits = self.seg_head(point.feat)
        offset_logits = self.offset_head(point.feat)
        
        # 根据第一模块的结果，进行点偏移和聚类，准备第二模块的输入
        # 偏移并聚类
        clustering = self.get_cluster(point,seg_logits=seg_logits,offset_logits=offset_logits)
        
            
            
        # 保存一个测试数据 
        if point['grid_coord'].get_device() == 0:
            data = {}
            data['grid_coord'] = point['grid_coord']
            data['offset'] = offset_logits
            data['segment'] = point['segment']
            data['clust'] = clustering.labels_
            data['batch_offset'] = point['offset']
            data['coord'] = point['coord']
            data['batch'] = point['batch']
            data['seg_logits'] = seg_logits
            torch.save(data,'testdata.pth')
        
        # train
        if self.training:
            
            # 计算loss seg
            loss_seg = self.criteria(seg_logits, input_dict["segment"])
            
    
            
            # 计算offset target
            offset_target = self.get_offset_target(point)
            
            # 计算loss offset
            loss_offset = self.get_loss_offset(offset_logits=offset_logits, offset_target=offset_target)
            print('val offset loss',loss_offset)
            
            # 计算dir loss
            # offset_pred_norm = torch.norm(offset_logits,dim=1).view(-1,1)
            # offset_pred_dir = torch.div(offset_logits, offset_pred_norm)
            # mask_pred = offset_pred_norm > 0.01
            # offset_target_norm = torch.norm(offset_target, dim=1).view(-1,1)
            # offset_target_dir = torch.div(offset_target, offset_target_norm)
            # mask_target = offset_target_norm > 0.01
            # mask = mask_pred & mask_target
            # mask = mask.squeeze()
            # dir_mat = torch.sum(offset_pred_dir * offset_target_dir, dim=1)
            # dir_mat = dir_mat - 1
            # dir_mat = dir_mat * dir_mat
            # dir_mat = dir_mat[mask]
            # loss_dir = torch.div(torch.sum(dir_mat),dir_mat.shape[0])
            loss_dir = self.get_loss_dir(offset_logits=offset_logits, offset_target=offset_target)
            print('train loss dir',loss_dir)
            
            
            # loss=loss_seg+loss_offset
            # return dict(loss=loss)
            return dict(loss_seg=loss_seg, loss_offset=loss_offset, loss_dir=loss_dir)
        # eval
        elif "segment" in input_dict.keys():
            
            # 计算loss seg
            loss_seg = self.criteria(seg_logits, input_dict["segment"])
     
            # 计算offset target
            offset_target = self.get_offset_target(point)
            
            # 计算loss offset
            loss_offset = self.get_loss_offset(offset_logits=offset_logits, offset_target=offset_target)
            print('val offset loss',loss_offset)
            
            # 计算dir loss
            # offset_pred_norm = torch.norm(offset_logits,dim=1).view(-1,1)
            # offset_pred_dir = torch.div(offset_logits, offset_pred_norm)
            # mask_pred = offset_pred_norm > 0.01
            # offset_target_norm = torch.norm(offset_target, dim=1).view(-1,1)
            # offset_target_dir = torch.div(offset_target, offset_target_norm)
            # mask_target = offset_target_norm > 0.01
            # mask = mask_pred & mask_target
            # mask = mask.squeeze()
            # dir_mat = torch.sum(offset_pred_dir * offset_target_dir, dim=1)
            # dir_mat = dir_mat - 1
            # dir_mat = dir_mat * dir_mat
            # dir_mat = dir_mat[mask]
            # loss_dir = torch.div(torch.sum(dir_mat),dir_mat.shape[0])
            loss_dir = self.get_loss_dir(offset_logits=offset_logits, offset_target=offset_target)
            print('val loss dir',loss_dir)
            # loss=loss_seg+loss_offset
            # return dict(loss=loss,seg_logits=seg_logits, offset_logits=offset_logits)
            return dict(loss_seg=loss_seg, loss_offset=loss_offset, loss_dir=loss_dir,seg_logits=seg_logits, offset_logits=offset_logits)
        # test
        else:
            return dict(seg_logits=seg_logits, offset_logits=offset_logits)
        
        
        
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
        for label in range(1, 17):
            mask = (segment == label)
            # print(mask.shape)
            center = torch.mean(grid_coord[mask], dim=0)
            offset_target[mask] = center - grid_coord[mask]
        return offset_target
    
    
    def get_cluster(self, point, seg_logits, offset_logits):
        batch_offset = point['offset']
        last_offset = 0
        num = 0
        if int(batch_offset.shape[0] / 8 )> 0:
            check_num = int(batch_offset.shape[0]/8)
        else:
            check_num = 1
        
        for b in range(check_num):
            # 取出该batch
            grid_coord_b = point['grid_coord'][last_offset:batch_offset[b]].cpu().detach().numpy()
            offset_logits_b = offset_logits[last_offset:batch_offset[b]].cpu().detach().numpy()
            segment_b = point['segment'][last_offset:batch_offset[b]].cpu().detach().numpy()
            mask = (segment_b != 0)
            moved_point = grid_coord_b + offset_logits_b
            moved_point = moved_point
            moved_point = moved_point[mask]
            
            # 计算聚类
            clustering = DBSCAN(eps=2, min_samples=150).fit(moved_point)
            print('clust on ',b,'聚类数',len(set(clustering.labels_)),'应有',len(set(segment_b)),'点数',np.count_nonzero(clustering.labels_ != -1),'应有',np.count_nonzero(segment_b != 0))
            
            # 计算聚类正确率
            label_to_cls = np.zeros(17) - 1
            seg_cls = np.zeros(segment_b[mask].shape)
            # 对于每个聚类，找到他们认为对应的标签
            for cls in range(len(set(clustering.labels_))-1):
                cls_mask = (clustering.labels_ == cls)
                counter = Counter(segment_b[mask][cls_mask])  
                most_common_element = counter.most_common(1)[0] 
                
                most ,count = most_common_element
                # print(most,count/np.count_nonzero(clustering.labels_ == cls))
                label = most
                label_to_cls[label] = cls
                seg_cls[np.where(clustering.labels_ == cls)] = label
            # 对于每个label，计算它们的正确率
            for label in np.unique(segment_b):
                print('类型',label, '应有',np.count_nonzero(segment_b == label), '实有',np.count_nonzero(clustering.labels_ == label_to_cls[label]),'其中正确',np.sum((segment_b[mask]==label)&(seg_cls == label)))
            print('聚类正确比例',(segment_b[mask] == seg_cls).sum()/ len(seg_cls))
            if len(set(clustering.labels_)) == len(set(segment_b)):
                print('聚类数正确')
                num += 1
            else:
                print('错误')
            
            last_offset = batch_offset[b]
        print('聚类数量正确率',num/batch_offset.shape[0])
        # 这个返回目前只为测试数据返回
        return clustering
"""
Implementation of Brady Zhou's cross view transformer
"""
import torch.nn as nn
from einops import rearrange
from opencood.models.sub_modules.cvt_modules import CrossViewModule
from opencood.models.backbones.resnet_ms import ResnetEncoder
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.fusion_modules.v2v_fuse import V2VNetFusion
from opencood.models.fuse_modules.v2v_fuse import V2VNetFusion as LidarFusion
import torch.nn.functional as F
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
import torch

class FusionV2VNet(nn.Module):
    def __init__(self, config):
        super(FusionV2VNet, self).__init__()
        self.max_cav = config['max_cav']
        # encoder params
        self.encoder = ResnetEncoder(config['encoder'])

        # cvm params
        cvm_params = config['cvm']
        cvm_params['backbone_output_shape'] = self.encoder.output_shapes
        self.cvm = CrossViewModule(cvm_params)

        # spatial feature transform module
        self.downsample_rate = config['sttf']['downsample_rate']
        self.discrete_ratio = config['sttf']['resolution']
        self.use_roi_mask = config['sttf']['use_roi_mask']

        # PIllar VFE
        self.pillar_vfe = PillarVFE(config['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=config['voxel_size'],
                                    point_cloud_range=config['lidar_range'])
        self.scatter = PointPillarScatter(config['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(config['base_bev_backbone'], 64)
        self.lidar_fusion_net = LidarFusion(config['v2vfusion'])

        # spatial fusion
        self.fusion_net = V2VNetFusion(config['v2vnet_fusion'])

        # decoder params
        decoder_params = config['decoder']
        # decoder for dynamic and static differet
        self.decoder = NaiveDecoder(decoder_params)

        self.target = config['target']
        self.cls_head = nn.Conv2d(128 * 2, config['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * config['anchor_number'],
                                  kernel_size=1)
        
        self.reduce_conv = nn.Conv2d(288, 256, kernel_size=1)

        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in config:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(config['shrink_header'])


    def forward(self, batch_dict):
        x = batch_dict['inputs']
        b, l, m, _, _, _ = x.shape

        # shape: (B, max_cav, 4, 4)
        pairwise_t_matrix = batch_dict['pairwise_t_matrix']
        record_len = batch_dict['record_len']

        x = self.encoder(x)
        batch_dict.update({'features': x})
        x = self.cvm(batch_dict)

        # B*L, C, H, W
        x = x.squeeze(1)
        # fuse all agents together to get a single bev map, b h w c
        x = self.fusion_net(x, record_len, pairwise_t_matrix, None)
        x = x.unsqueeze(1).permute(0, 1, 4, 2, 3)

        voxel_features = batch_dict['processed_lidar']['voxel_features']
        voxel_coords = batch_dict['processed_lidar']['voxel_coords']
        voxel_num_points = batch_dict['processed_lidar']['voxel_num_points']
        record_len = batch_dict['record_len']

        pairwise_t_matrix = batch_dict['pairwise_t_matrix']

        lidar_batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        lidar_batch_dict = self.pillar_vfe(lidar_batch_dict)
        # n, c -> N, C, H, W
        lidar_batch_dict = self.scatter(lidar_batch_dict)
        lidar_batch_dict = self.backbone(lidar_batch_dict)

        spatial_features_2d = lidar_batch_dict['spatial_features_2d']
        
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        fused_feature = self.lidar_fusion_net(spatial_features_2d,
                                        record_len,
                                        pairwise_t_matrix)

        # dynamic head
        x = self.decoder(x)
        x = rearrange(x, 'b l c h w -> (b l) c h w')
        # L = 1 for sure in intermedaite fusion at this point
        x = F.interpolate(x, size=(50, 176), mode='bilinear', align_corners=False)
        x = torch.cat([x, fused_feature], dim=1) 
        x = self.reduce_conv(x)  # (B, 256, 50, 176)

        b = x.shape[0]
        psm = self.cls_head(x)
        rm = self.reg_head(x)

        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict

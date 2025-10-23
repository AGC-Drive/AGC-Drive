# Author: Binyu Zhao <byzhao@stu.hit.edu.cn>
# Author: Yue Hu <18671129361@sjtu.edu.cn>

from re import A
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18
import torch.nn.functional as F
from opencood.utils.camera_utils import bin_depths


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)  # 上采样 BxCxHxW->BxCx2Hx2W

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

        #x1 = self.up(x1)
        #x1 = torch.cat([x2, x1], dim=1)
        #return self.conv(x1)


class ImgCamEncode(nn.Module):
    def __init__(self, D, C, downsample, ddiscr, mode, use_gt_depth=False, depth_supervision=True):
        super(ImgCamEncode, self).__init__()
        self.D = D  # 42
        self.C = C  # 64
        self.downsample = downsample
        self.d_min = ddiscr[0]
        self.d_max = ddiscr[1]
        self.num_bins = ddiscr[2]
        self.mode = mode
        self.use_gt_depth = use_gt_depth
        self.depth_supervision = depth_supervision # in the case of not use gt depth
        self.chain_channels = 256 # 512

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")
        print("Number of parameter EfficientNet: %d" % (sum([param.nelement() for param in self.trunk.parameters()])))

        self.up1 = Up(320+112, self.chain_channels)
        if downsample == 8:
            self.up2 = Up(self.chain_channels+40, self.chain_channels)
        if not use_gt_depth:
            self.depth_head = nn.Conv2d(self.chain_channels, self.D, kernel_size=1, padding=0)
        self.image_head = nn.Conv2d(self.chain_channels, self.C, kernel_size=1, padding=0)

    def get_eff_features(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))  #  x: 24 x 32 x 64 x 176
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        if self.downsample == 8:
            x = self.up2(x, endpoints['reduction_3'])
        return x


    def forward(self, x, depth_maps, record_len):
        _, _, oriH, oriW = x.shape
        cum_sum_len = torch.cumsum(record_len, dim=0)

        if depth_maps.dim() == 5:
            # B, T, N, H, W -> B, N, H, W
            depth_map = depth_maps[:,0,:,:,:]
            
            B, T, N, _, _ = depth_maps.shape
            assert T == 2, f"T={T} should be 2, which first for self-image and second for ego-image"

            ego_index = 0
            # get fused depth map for ego agent
            for next_ego_index in cum_sum_len:
                maps_for_ego = depth_maps[ego_index: next_ego_index,1,:,:,:]    # size= [sum(cav), num(camera), H, W]
                max_value = torch.max(maps_for_ego)
                maps_for_ego[maps_for_ego<0] = max_value + 1
                maps_for_ego, _ = torch.min(maps_for_ego, dim=0)
                maps_for_ego[maps_for_ego>max_value] = -1

                ego_depth_mask = ((maps_for_ego[0]) > 0).long() # size= [num(camera), H, W]
                # torch.count_nonzero(), tensor.numel()
                depth_map[ego_index] = depth_map[ego_index]*ego_depth_mask + maps_for_ego*(1-ego_depth_mask)

                # update index
                ego_index += next_ego_index
        else:
            # B, N, H, W
            depth_map = depth_maps
        
        x_img = x[:,:3:,:,:]    # origin x: (B*num(cav), C, H, W)
        features = self.get_eff_features(x_img)     # 8x downscale feature: (B*num(cav), set_channels(e.g.256), H/4, W/4)
        x_img = self.image_head(features) #  8x downscale feature: B*N x C x fH x fW(24 x 64 x 8 x 22). C is the channel for next stage (i.e. bev)

        # resize depth
        batch, _, h, w = features.shape
        max_value = torch.max(depth_map)
        depth_map[depth_map<0] = max_value + 1        
        if oriH % h == 0 and oriW % w == 0:
            scaleh, scalew = oriH // h, oriW // w
            pool_layer = nn.MaxPool2d(kernel_size=(scaleh, scalew), stride=(scaleh, scalew))
        else:
            pool_layer = nn.AdaptiveMaxPool2d((h, w), return_indices=False)
        depth_map = -1 * pool_layer(-1 * depth_map)
        depth_map[depth_map>max_value] = 0

        # generate one-hot refered ground truth
        depth_mask = ((depth_map) > 0).long().reshape(-1, 1, h, w)
        depth_map = depth_map.to(torch.int64).flatten(2).squeeze(1)
        one_hot_depth_map = []
        for batch_map in depth_map:
            one_hot_depth_map.append(F.one_hot(batch_map, num_classes=self.D))
        one_hot_depth_map = torch.stack(one_hot_depth_map).reshape(batch, h, w, self.D).permute(0,3,1,2) # [B*N, num_bins, fH, fW]

        depth_score = self.depth_head(features)
        depth_pred = F.softmax(depth_score, dim=1)

        final_depth = depth_mask * one_hot_depth_map + (1-depth_mask) * depth_pred

        # size: final_depth=[B*num(cav), D, H, W]; x_img=[B*num(cav), C, H, W]
        new_x = final_depth.unsqueeze(1) * x_img.unsqueeze(2)
        #nodepth_x = depth_pred.unsqueeze(1) * x_img.unsqueeze(2)
        return x_img, new_x#, nodepth_x

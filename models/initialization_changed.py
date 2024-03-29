import torch
import torch.nn as nn
import torch.nn.functional as F
from .FE import BasicConv2d
import pdb
from .submodules import BuildVolume2d


class INIT(nn.Module):
    """
    Tile hypothesis initialization
    input: dual feature pyramid
    output: initial tile hypothesis pyramid
    """
    def __init__(self, args):
        super().__init__()
        self.maxdisp = args.maxdisp
        fea_c1x = args.fea_c[4]
        fea_c2x = args.fea_c[3]
        fea_c4x = args.fea_c[2]
        fea_c8x = args.fea_c[1]
        fea_c16x = args.fea_c[0]
        self.tile_conv1x = nn.Sequential(
            BasicConv2d(fea_c1x, fea_c1x, 4, 4, 0, 1),
            nn.Conv2d(fea_c1x, fea_c1x, 1, 1, 0, bias=False)
        )

        self.tile_conv2x = nn.Sequential(
            BasicConv2d(fea_c2x, fea_c2x, 4, 4, 0, 1),
            nn.Conv2d(fea_c2x, fea_c2x, 1, 1, 0, bias=False)
        )

        self.tile_conv4x = nn.Sequential(
            BasicConv2d(fea_c4x, fea_c4x, 4, 4, 0, 1),
            nn.Conv2d(fea_c4x, fea_c4x, 1, 1, 0, bias=False)
        )

        self.tile_conv8x = nn.Sequential(
            BasicConv2d(fea_c8x, fea_c8x, 4, 4, 0, 1),
            nn.Conv2d(fea_c8x, fea_c8x, 1, 1, 0, bias=False)
        )

        self.tile_conv16x = nn.Sequential(
            BasicConv2d(fea_c16x, fea_c16x, 4, 4, 0, 1),
            nn.Conv2d(fea_c16x, fea_c16x, 1, 1, 0, bias=False)
        )

        self.tile_fea_dscrpt16x = BasicConv2d(fea_c16x+1, 13, 1, 1, 0, 1)
        self.tile_fea_dscrpt8x = BasicConv2d(fea_c8x+1, 13, 1, 1, 0, 1)
        self.tile_fea_dscrpt4x = BasicConv2d(fea_c4x+1, 13, 1, 1, 0, 1)
        self.tile_fea_dscrpt2x = BasicConv2d(fea_c2x+1, 13, 1, 1, 0, 1)
        self.tile_fea_dscrpt1x = BasicConv2d(fea_c1x+1, 13, 1, 1, 0, 1)

        self._build_volume_2d16x = BuildVolume2d(self.maxdisp//16)
        self._build_volume_2d8x = BuildVolume2d(self.maxdisp//8)
        self._build_volume_2d4x = BuildVolume2d(self.maxdisp//4)
        self._build_volume_2d2x = BuildVolume2d(self.maxdisp//2)
        self._build_volume_2d1x = BuildVolume2d(self.maxdisp)
        
        # change width of fea_r equal to fea_l (/4)
        self._fea_r2l_1x = nn.Conv2d(fea_c1x , fea_c1x ,(1,4), (1,4))
        self._fea_r2l_2x = nn.Conv2d(fea_c2x , fea_c2x ,(1,4), (1,4))
        self._fea_r2l_4x = nn.Conv2d(fea_c4x , fea_c4x ,(1,4), (1,4))
        self._fea_r2l_8x = nn.Conv2d(fea_c8x , fea_c8x ,(1,4), (1,4))
        self._fea_r2l_16x = nn.Conv2d(fea_c16x, fea_c16x,(1,4), (1,4))

    def tile_features(self, fea_l, fea_r):

        right_fea_pad = [0, 3, 0, 0]
        # pdb.set_trace()

        tile_fea_l1x = self.tile_conv1x(fea_l[-1])
        padded_fea_r1x = F.pad(fea_r[-1], right_fea_pad)
        self.tile_conv1x[0][0].stride = (4, 1)
        tile_fea_r1x = self.tile_conv1x(padded_fea_r1x)
        self.tile_conv1x[0][0].stride = (4, 4)

        tile_fea_l2x = self.tile_conv2x(fea_l[-2])
        padded_fea_r2x = F.pad(fea_r[-2], right_fea_pad)
        self.tile_conv2x[0][0].stride = (4, 1)
        tile_fea_r2x = self.tile_conv2x(padded_fea_r2x)
        self.tile_conv2x[0][0].stride = (4, 4)

        tile_fea_l4x = self.tile_conv4x(fea_l[-3])
        padded_fea_r4x = F.pad(fea_r[-3], right_fea_pad)
        self.tile_conv4x[0][0].stride = (4, 1)
        tile_fea_r4x = self.tile_conv4x(padded_fea_r4x)
        self.tile_conv4x[0][0].stride = (4, 4)

        tile_fea_l8x = self.tile_conv8x(fea_l[-4])
        padded_fea_r8x = F.pad(fea_r[-4], right_fea_pad)
        self.tile_conv8x[0][0].stride = (4, 1)
        tile_fea_r8x = self.tile_conv8x(padded_fea_r8x)
        self.tile_conv8x[0][0].stride = (4, 4)

        tile_fea_l16x = self.tile_conv16x(fea_l[-5])
        padded_fea_r16x = F.pad(fea_r[-5], right_fea_pad)
        self.tile_conv16x[0][0].stride = (4, 1)
        tile_fea_r16x = self.tile_conv16x(padded_fea_r16x)
        self.tile_conv16x[0][0].stride = (4, 4)

        return [
            [tile_fea_l16x, tile_fea_r16x],
            [tile_fea_l8x, tile_fea_r8x],
            [tile_fea_l4x, tile_fea_r4x],
            [tile_fea_l2x, tile_fea_r2x],
            [tile_fea_l1x, tile_fea_r1x],
        ]

    def tile_hypothesis_pyramid(self, tile_feature_pyramid):
        init_tile_cost16x = self._build_volume_2d16x(tile_feature_pyramid[0][0], tile_feature_pyramid[0][1])
        init_tile_cost8x = self._build_volume_2d8x(tile_feature_pyramid[1][0], tile_feature_pyramid[1][1])
        init_tile_cost4x = self._build_volume_2d4x(tile_feature_pyramid[2][0], tile_feature_pyramid[2][1])
        init_tile_cost2x = self._build_volume_2d2x(tile_feature_pyramid[3][0], tile_feature_pyramid[3][1])
        init_tile_cost1x = self._build_volume_2d1x(tile_feature_pyramid[4][0], tile_feature_pyramid[4][1])

        min_tile_cost16x, min_tile_disp16x = torch.min(init_tile_cost16x, 1)
        min_tile_cost8x, min_tile_disp8x = torch.min(init_tile_cost8x, 1)
        min_tile_cost4x, min_tile_disp4x = torch.min(init_tile_cost4x, 1)
        min_tile_cost2x, min_tile_disp2x = torch.min(init_tile_cost2x, 1)
        min_tile_cost1x, min_tile_disp1x = torch.min(init_tile_cost1x, 1)

        min_tile_cost16x = torch.unsqueeze(min_tile_cost16x, 1)
        min_tile_cost8x = torch.unsqueeze(min_tile_cost8x, 1)
        min_tile_cost4x = torch.unsqueeze(min_tile_cost4x, 1)
        # min_tile_cost2x = torch.unsqueeze(min_tile_cost2x, 1)
        # min_tile_cost1x = torch.unsqueeze(min_tile_cost1x, 1)

        min_tile_disp16x = min_tile_disp16x.float().unsqueeze(1)
        min_tile_disp8x = min_tile_disp8x.float().unsqueeze(1)
        min_tile_disp4x = min_tile_disp4x.float().unsqueeze(1)
        min_tile_disp2x = min_tile_disp2x.float().unsqueeze(1)
        min_tile_disp1x = min_tile_disp1x.float().unsqueeze(1)
        
        # diff: when disp=0 and disp=max_disp/2
        fea_r_equal16x = self._fea_r2l_16x(tile_feature_pyramid[0][1])
        fea_r_equal8x = self._fea_r2l_8x(tile_feature_pyramid[1][1])
        fea_r_equal4x = self._fea_r2l_4x(tile_feature_pyramid[2][1])
        fea_r_equal2x = self._fea_r2l_2x(tile_feature_pyramid[3][1])
        fea_r_equal1x = self._fea_r2l_1x(tile_feature_pyramid[4][1])
        
        # warp_m2_16x = self.warp(fea_r_equal16x, self.maxdisp/2/16)
        # warp_m2_8x = self.warp(fea_r_equal8x , self.maxdisp/2/8)
        # warp_m2_4x = self.warp(fea_r_equal4x , self.maxdisp/2/4)
        warp_m2_2x = self.warp(fea_r_equal2x , self.maxdisp/2/2)
        warp_m2_1x = self.warp(fea_r_equal1x , self.maxdisp/2)
        
        diff_0_16x = torch.unsqueeze(torch.norm(fea_r_equal16x - tile_feature_pyramid[0][0], 1, 1), 1)
        diff_0_8x = torch.unsqueeze(torch.norm(fea_r_equal8x  - tile_feature_pyramid[1][0], 1, 1), 1)
        diff_0_4x = torch.unsqueeze(torch.norm(fea_r_equal4x  - tile_feature_pyramid[2][0], 1, 1), 1)
        diff_0_2x = torch.unsqueeze(torch.norm(fea_r_equal2x  - tile_feature_pyramid[3][0], 1, 1), 1)
        diff_0_1x = torch.unsqueeze(torch.norm(fea_r_equal1x  - tile_feature_pyramid[4][0], 1, 1), 1)
        
        # diff_m2_16x = torch.unsqueeze(torch.norm(warp_m2_16x - tile_feature_pyramid[0][0], 1, 1), 1)
        # diff_m2_8x = torch.unsqueeze(torch.norm(warp_m2_8x - tile_feature_pyramid[1][0], 1, 1), 1)
        # diff_m2_4x = torch.unsqueeze(torch.norm(warp_m2_4x - tile_feature_pyramid[2][0], 1, 1), 1)
        # diff_m2_2x = torch.unsqueeze(torch.norm(warp_m2_2x - tile_feature_pyramid[3][0], 1, 1), 1)
        # diff_m2_1x = torch.unsqueeze(torch.norm(warp_m2_1x - tile_feature_pyramid[4][0], 1, 1), 1)
        
        # tile_dscrpt
        tile_dscrpt16x = self.tile_fea_dscrpt16x(torch.cat([min_tile_cost16x, tile_feature_pyramid[0][0]], 1))
        tile_dscrpt8x = self.tile_fea_dscrpt8x(torch.cat([min_tile_cost8x, tile_feature_pyramid[1][0]], 1))
        tile_dscrpt4x = self.tile_fea_dscrpt4x(torch.cat([min_tile_cost4x, tile_feature_pyramid[2][0]], 1))

        # tile_dscrpt16x = self.tile_fea_dscrpt16x(torch.cat([diff_m2_16x, tile_feature_pyramid[0][0]], 1))
        # tile_dscrpt8x = self.tile_fea_dscrpt8x(torch.cat([diff_m2_8x, tile_feature_pyramid[1][0]], 1))
        # tile_dscrpt4x = self.tile_fea_dscrpt4x(torch.cat([diff_m2_4x, tile_feature_pyramid[2][0]], 1))
        tile_dscrpt2x = self.tile_fea_dscrpt2x(torch.cat([min_tile_disp2x, tile_feature_pyramid[3][0]], 1))
        tile_dscrpt1x = self.tile_fea_dscrpt1x(torch.cat([min_tile_disp1x, tile_feature_pyramid[4][0]], 1))

        tile_dx16x = torch.zeros_like(diff_0_16x)
        tile_dx8x = torch.zeros_like(diff_0_8x)
        tile_dx4x = torch.zeros_like(diff_0_4x)
        tile_dx2x = torch.zeros_like(diff_0_2x)
        tile_dx1x = torch.zeros_like(diff_0_1x)

        tile_dy16x = torch.zeros_like(diff_0_16x)
        tile_dy8x = torch.zeros_like(diff_0_8x)
        tile_dy4x = torch.zeros_like(diff_0_4x)
        tile_dy2x = torch.zeros_like(diff_0_2x)
        tile_dy1x = torch.zeros_like(diff_0_1x)
        # pdb.set_trace()

        # tile_hypothesis
        tile_hyp16x = torch.cat([min_tile_disp16x, tile_dx16x, tile_dy16x, tile_dscrpt16x], 1)
        tile_hyp8x = torch.cat([min_tile_disp8x, tile_dx8x, tile_dy8x, tile_dscrpt8x], 1)
        tile_hyp4x = torch.cat([min_tile_disp4x, tile_dx4x, tile_dy4x, tile_dscrpt4x], 1)

        # tile_hyp16x = torch.cat([diff_0_16x, tile_dx16x, tile_dy16x, tile_dscrpt16x], 1)
        # tile_hyp8x = torch.cat([diff_0_8x, tile_dx8x, tile_dy8x, tile_dscrpt8x ], 1)
        # tile_hyp4x = torch.cat([diff_0_4x, tile_dx4x, tile_dy4x, tile_dscrpt4x ], 1)
        tile_hyp2x = torch.cat([diff_0_2x, tile_dx2x, tile_dy2x, tile_dscrpt2x ], 1)
        tile_hyp1x = torch.cat([diff_0_1x, tile_dx1x, tile_dy1x, tile_dscrpt1x ], 1)

        return [
            [
                init_tile_cost16x,
                init_tile_cost8x,
                init_tile_cost4x,
                init_tile_cost2x,
                init_tile_cost1x,
            ],
            [
                tile_hyp16x,
                tile_hyp8x,
                tile_hyp4x,
                tile_hyp2x,
                tile_hyp1x,
            ]
        ]

    def warp(self, x, disp):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W, device=x.device).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device=x.device).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        vgrid = torch.cat((xx, yy), 1).float()

        # vgrid = Variable(grid)
        vgrid[:,:1,:,:] = vgrid[:,:1,:,:] - disp

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid)
        return output
    
    def forward(self, fea_l_pyramid, fea_r_pyramid):
        tile_feature_duo_pyramid = self.tile_features(fea_l_pyramid, fea_r_pyramid)
        init_cv_pyramid, init_hypo_pyramid = self.tile_hypothesis_pyramid(tile_feature_duo_pyramid)
        return [init_cv_pyramid, init_hypo_pyramid]








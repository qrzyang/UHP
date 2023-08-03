from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import hessian

def disp_to_depth(disp, min_depth = 1, max_depth = 100):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    disp = disp/192.0
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth
def disp_to_depth_real(disp):
    focallength = 720.
    baseline = 0.54
    
    disp_cut = torch.clamp(disp,min=0.1)
    depth = focallength*baseline/disp_cut
    return depth

def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class Backproject_PL_Loss(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(Backproject_PL_Loss, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)
        
    def cal_grad(self, points):
        points = F.pad(points, (1,1,1,1),'constant',0)
        # x - direction
        dis_x = torch.sqrt((points[:, 0, :, 1:] - points[:, 0, :, :-1]).pow(2) + (points[:, 1, :, 1:] - points[:, 1, :, :-1]).pow(2) + 1e-6) # distance
        dis_x.detach_()
        dz_x = (points[:, 2, :, 1:] - points[:, 2, :, :-1]) / dis_x
        dz_x2 = (dz_x[..., 1:] - dz_x[..., :-1]) / dis_x[..., :-1]
        
        # y - direction
        dis_y = torch.sqrt((points[:, 0, 1:, :] - points[:, 0, :-1, :]).pow(2) + (points[:, 1, 1:, :] - points[:, 1, :-1, :]).pow(2) + 1e-6)  # distance
        dis_y.detach_()
        dz_y = (points[:, 2, 1:, :] - points[:, 2, :-1, :]) / dis_y     
        dz_y2 = (dz_y[..., 1:, :] - dz_y[..., :-1, :]) / dis_y[..., :-1, :]
        
        dz_xy = (dz_x[..., 1:, :] - dz_x[..., :-1, :]) / dis_y[..., :, :-1]

        dz_yx = (dz_y[..., 1:] - dz_y[..., :-1]) / dis_x[..., :-1, :]
        
        grad_2order = torch.abs(dz_x2)[..., 1:-1,:] + 0.5*torch.abs(dz_y2)[..., 1:-1] + torch.abs(dz_xy)[...,:-1,:-1] + torch.abs(dz_yx)[...,:-1,:-1]
        return grad_2order.unsqueeze(1)

    def forward(self, disp, inv_K, edge, **pad):
        # _, depth = disp_to_depth(disp.clone())
        depth = disp_to_depth_real(disp.clone())
        _,_,h,w = disp.size()
            
        shift_tmp = self.width-depth.shape[-1]
        depth = F.pad(depth, (shift_tmp,0,0,0),'replicate')
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)
        
        cam_points = cam_points.view(self.batch_size,-1,depth.shape[-2],depth.shape[-1])

        cam_points = cam_points[...,shift_tmp:]
        grad_2order = self.cal_grad(cam_points) * 1e-2
        grad_2order = torch.tanh(grad_2order)   
        
        edge_cut = torch.where(edge<0.5, torch.zeros_like(edge), edge)
        edge_log = (-torch.log((1-edge_cut) + 1e-8))
        edge_log = F.pad(edge_log[..., 1:-1, 1:-1], (1,1,1,1),'constant',0)
        Pl_loss = grad_2order * edge_log
        
        return cam_points, grad_2order.clone(), Pl_loss.mean(), depth, Pl_loss.clone()


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


def get_smooth_loss(disp, img, mask = None):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    mask: 0-1 tensor, ignore the 0 pixels
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    if mask is not None:
        grad_disp_x = grad_disp_x[mask[:, :, :, :grad_disp_x.shape[3]]==1]
        grad_disp_y = grad_disp_y[mask[:, :, :grad_disp_y.shape[2], :]==1]

    return grad_disp_x.mean() + grad_disp_y.mean()

def get_Pl_loss(disp, edge):
    """Computes the smoothness loss for a disparity image
    The edge image is used for edge-aware smoothness
    """
    def dxx(img):
        return 2*(img[:,:,:-2,2:] + img[:,:,:-2,:-2] - 2*img[:,:,:-2,1:-1])
    
    def dyy(img):
        return 2*(img[:,:,2:,:-2] + img[:,:,:-2,:-2] - 2*img[:,:,1:-1,:-2])
    
    def dxy(img):
        return (img[:,:,1:-1,1:-1] + img[:,:,:-2,:-2] - img[:,:,1:-1,:-2] - img[:,:,:-2,1:-1])

    # grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    # grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
    
    # grad_disp = grad_disp_x[:,:,:-1,:] + grad_disp_y[:,:,:,:-1]
    grad_disp = torch.abs(dxx(disp)) + torch.abs(dyy(disp)) + torch.abs(dxy(disp))

    edge_cut = torch.where(edge<0.5, torch.zeros_like(edge), edge)
    grad_disp *= -torch.log((1-edge_cut[:,:,:-2,:-2]) + 1e-8)

    return grad_disp.mean()

class PL_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.laplacian_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, padding_mode='replicate',bias=False)
        self.laplacian_conv.weight.data = kernel
        
    def forward(self, disp, edge):
        _, depth = disp_to_depth(disp)

        grad_depth = self.laplacian_conv(depth)
        grad_depth = torch.abs(grad_depth)
        grad_depth = F.pad(grad_depth[..., 1:-1,1:-1],(1, 1, 1, 1),mode='constant',value=0)

        edge_cut = torch.where(edge<0.5, torch.zeros_like(edge), edge)
        grad_depth_loss = grad_depth * (-torch.log((1-edge_cut[:,:,:,:]) + 1e-8))
        
        tmp_grad_graph = grad_depth                                 # TODO:DE
        tmp_max = torch.max(tmp_grad_graph)
        tmp_grad_graph = tmp_grad_graph / tmp_max
        tmp_grad_graph[tmp_grad_graph>0.1] = 0.
        
        tmp_depth = depth
        return grad_depth_loss.mean(), tmp_grad_graph, tmp_depth
        

def get_Pl_loss1(disp, edge):
    kernel = torch.tensor([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], dtype=torch.float32, device=disp.device).view(1, 1, 3, 3)

    # _, depth = disp_to_depth(disp)

    disp_pad = F.pad(disp, (1, 1, 1, 1), mode='constant', value=0)
    grad_disp = F.conv2d(disp_pad, kernel, padding=0).abs()

    edge_cut = torch.where(edge<0.5, torch.zeros_like(edge), edge)
    edge_log = (-torch.log((1-edge_cut) + 1e-8) + 1e-6)
    edge_log = F.pad(edge_log[..., 1:-1, 1:-1], (1,1,1,1),'constant',0)
    
    grad_disp_loss = grad_disp * edge_log

    return grad_disp_loss.mean(), grad_disp.clone().detach(), grad_disp_loss.clone().detach()



class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

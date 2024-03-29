from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from .utils import *
from .kitti_utils import *
from .layers import *
import argparse

import scipy
from .torchinterp1d import interp1d



class Loss(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.opt = self.get_opt()


        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.device = self.opt.device
        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"


        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.relu = torch.nn.ReLU()
        self.get_conf_loss = cluster_loss(self.opt)



        self.ssim = SSIM()
        self.ssim.to(self.device)


        self.backproject_PL_Loss = {}
        self.project_3d = {}
        for scale in [0]:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_PL_Loss[scale] = Backproject_PL_Loss(self.opt.batch_size, h, w)
            self.backproject_PL_Loss[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        # self.save_opts()





    def forward(self, inputs, outputs):

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses



    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.opt.device)
        scale_list = [k[1] for k in outputs.keys() if k.__contains__('disp')]
        for scale in scale_list:
            shift_pixels = inputs['shift_pixels']
            disp = outputs[("disp", scale)][..., :-shift_pixels]
            source_scale = 0

            outputs[("disp", scale)]=disp
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                shift_disp = F.pad(disp, (shift_pixels,0, 0, 0))
                warp_right = self.warp(
                    inputs[("color", frame_id, source_scale)],
                    shift_disp)
                outputs[("color", frame_id, scale)] = warp_right[..., shift_pixels:]
                outputs["warped"] = outputs[("color", frame_id, scale)].clone().detach()



    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        
        return reprojection_loss
    
    def compute_reprojection_loss_Tmp(self, pred, target, outputs):     # TODO:DE
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        reprojection_loss = l1_loss
        ssim_loss = self.ssim(pred, target).mean(1, True)
            
        outputs["ssim_loss"],  outputs["l1_loss"] = ssim_loss, l1_loss
        return

    def compute_rep_crop_loss(self, pred, target):
        abs_diff = torch.abs(target - pred)
        # l1_loss = abs_diff.mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)


    def compute_losses(self, inputs, outputs):
        """Compute the reprojection, smoothness and proxy supervised losses for a minibatch
        """
        losses = {}
        total_loss = 0
        global_loss_weights = self.opt.loss_weights
        warp_loss_weights = [1/64, 1/32, 1/32, 1/16, 1/16, 1/8, 1/8, 1/8, 1/4, 1/4, 1/2, 1]
        prop_diff_pyramid = []

        ignore_mask = torch.ones_like(outputs[("disp", 0)])  # False - ignore

        scale_list = [k[1] for k in outputs.keys() if k.__contains__('disp')]


        for scale in scale_list:
        # for scale in [11]:
            loss = 0
            reprojection_losses = []

            source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, source_scale)]
            target = inputs[("color", 0, source_scale)]


            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
                self.compute_reprojection_loss_Tmp(pred, target, outputs)               # TODO:DE
                
                diff = torch.abs(pred - target).mean(1) # diff use the mean of color diff
                diff = diff.unsqueeze(1)
                prop_diff_pyramid.append(diff)


            reprojection_losses = torch.cat(reprojection_losses, 1)

            reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)

            if scale>7 and global_loss_weights[4]==1:
                with torch.no_grad():
                    def get_left_edge(disp):
                        B, C, H, W = disp.size()
                        # mesh grid
                        xx = torch.arange(0, W, device=disp.device).view(1, -1).repeat(H, 1)
                        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
                        xx = xx - disp + inputs["shift_pixels"]
                        edge_bool = xx<=1
                        tmp_edge_bool = edge_bool
                        if torch.any(edge_bool):
                            for i in range((int)(xx[edge_bool].max())):
                                tmp_edge_bool = F.pad(tmp_edge_bool[...,1:], (0, 1, 0, 0), mode='constant', value=0)
                                edge_bool = tmp_edge_bool | edge_bool
                        return edge_bool
                        
                    # occlusion 
                    grad_x = disp[:, :, :, 1:] - disp[:, :, :, :-1]
                    grad_x = F.pad(grad_x, (1, 0, 0, 0), mode='constant', value=0)
                    grad_flag = grad_x>0.5    # TODO: constant value
                    disp_marked = torch.where(grad_flag, disp.clone(), torch.zeros_like(disp))
                    occ_tensor = torch.zeros_like(disp)
                    disp_marked1 = disp_marked.clone()
                    for occ_i in range((int)(torch.max(disp_marked))):
                        tmp = (disp_marked>0).to(torch.float)
                        occ_tensor += tmp
                        disp_marked = F.pad(disp_marked[...,1:], (0, 1, 0, 0), mode='constant', value=0)
                        disp_marked -= 1.01
                    occ_tensor = (occ_tensor>0).to(torch.float)
                        
                    # grad_flag = grad_x<-1    # TODO: constant value
                    # disp_marked = torch.where(grad_flag, disp.clone(), torch.zeros_like(disp))
                    # for occ_i in range((int)(torch.max(disp_marked))):
                    #     disp_marked = F.pad(disp_marked[...,1:], (0, 1, 0, 0), mode='constant', value=0)
                    #     disp_marked -= 5
                    #     tmp = (disp_marked>0).to(torch.float)
                    #     occ_tensor -= tmp
                    # occ_tensor = (occ_tensor>0).to(torch.float)
                        
                    # find left pixel of occ
                    tmp_occ_bool = (occ_tensor>0).to(torch.float)
                    left_pixel_bool = (tmp_occ_bool-F.pad(tmp_occ_bool[...,1:], (0, 1, 0, 0)))==-1
                    disp_left_pixel = torch.where(left_pixel_bool, disp.clone(), torch.zeros_like(disp)) 
                    
                    occ_tensor = tmp_occ_bool
                    for occ_i in range((int)(torch.max(disp_left_pixel))):
                        disp_left_pixel = F.pad(disp_left_pixel[...,:-1], (1, 0, 0, 0))
                        disp_left_pixel -= 1.4
                        tmp = (disp_left_pixel>0).to(torch.float)
                        occ_tensor -= tmp
                    # for occ_i in range(np.random.randint(10, max((int)(torch.max(disp_marked1)),11))):
                    #     disp_marked1 = F.pad(disp_marked1[...,:-1], (1, 0, 0, 0), mode='constant', value=0)
                    #     disp_marked1 -= 1.01
                    #     tmp = (disp_marked1>0).to(torch.float)
                    #     occ_tensor += tmp
                                 
                    left_edge = get_left_edge(disp) #TODO: check the grad
                    occ_tensor = torch.where((occ_tensor>0) | (left_edge), torch.zeros_like(occ_tensor), torch.ones_like(occ_tensor))
                    # occ_tensor[occ_tensor>0] = 0.
                    # occ_tensor[occ_tensor<=0] = 1
                    occ_tensor = occ_tensor.detach()
                    
                    # if True:
                    #     occ_array = []
                    #     disp_occ_tmp = disp.clone()
                    #     for occ_i in range(min((int)(torch.max(disp)), 80)):
                    #         disp_occ_tmp = F.pad(disp_occ_tmp[...,1:], (0, 1, 0, 0), mode='constant', value=0)
                    #         disp_occ_tmp -= 1
                    #         occ_array += [disp_occ_tmp]
                    #     occ_tensor = torch.cat(occ_array, 1)
                    #     occ_tensor,_ = torch.max(occ_tensor,dim=1,keepdim=True)
                    #     occ = occ_tensor - disp
                    #     disp_ref_tmp = occ + 2
                    #     disp_ref_tmp[occ<-0.9] = 0
                    #     disp_ref = self.warp(disp, disp_ref_tmp, padding_mode='reflection')
                    #     # disp_ref = torch.where(occ>=0, disp_ref_tmp, disp)
                    #     disp_ref.detach_()
                # outputs['occ'] = (occ>=0).to(torch.float)
                # if True:
                #     outputs['occ'] = torch.abs(disp - disp_ref)
                #     occ_ref_loss = torch.abs(disp - disp_ref)
                #     occ_ref_loss = occ_ref_loss.mean()* warp_loss_weights[scale] * global_loss_weights[2] #TODO: loss_weight
                #     losses[f"occ_ref_loss{scale}"] = occ_ref_loss
                #     outputs[f'occ_ref_{scale}'] = disp_ref
                reprojection_loss *= occ_tensor   # TODO:occ
                outputs['occ'] = (occ_tensor==0.).to(torch.float)
                # outputs[f'reprojection_loss_tensor_{scale}'] = reprojection_loss

            

            reprojection_loss = reprojection_loss.mean()
            # reprojection_loss *= ignore_mask
            # reproj_loss_top = torch.topk(reprojection_loss.view(-1), (int)(reprojection_loss.numel()/20))[0].min()
            # reproj_loss_high = (reprojection_loss>reproj_loss_top)
            # reprojection_loss*=((~reproj_loss_high).to(torch.float)+0.01)

            # reprojection_loss = reprojection_loss.sum() / ((ignore_mask).sum() + 1e-6)

            losses['reproj_loss/{}'.format(scale)] = reprojection_loss*warp_loss_weights[scale] * global_loss_weights[2]


            loss += reprojection_loss*warp_loss_weights[scale] * global_loss_weights[2]

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            
            # if scale > 10:
            #     smooth_loss = get_smooth_loss(norm_disp, color, ignore_mask) * warp_loss_weights[scale] * global_loss_weights[0] * 1e-2
            #     loss += smooth_loss
            #     losses[f"loss_smooth{scale}"] = smooth_loss
                
            # 3D Planar Segmentation Loss
            Pl_loss = 0
            if scale > 10:            
                # _, _, outputs["depth"] = self.Pl_loss(disp, inputs["edge"])
                # Pl_loss, outputs["Pl_loss"], outputs["depth"] = self.Pl_loss(disp, inputs["edge"])
                # Pl_loss, outputs["grad2_map"],outputs["Pl_loss"] = get_Pl_loss1(disp, inputs["edge"])
                # # Pl_loss = get_Pl_loss(disp, inputs["edge"]) + 0.001 * smooth_loss
                # outputs["Pl_loss"] = get_grad_disp(disp, inputs["edge"])
                outputs["points"], outputs["grad2_map"], Pl_loss, outputs["depth"], outputs["Pl_loss"] = self.backproject_PL_Loss[0](disp,inputs[("inv_K", 0)], inputs["edge"])
                losses[f"loss_PL{scale}"] = Pl_loss * warp_loss_weights[scale] * global_loss_weights[3]

            loss += Pl_loss * warp_loss_weights[scale] * global_loss_weights[3]


            # # mssim loss
            # for frame_id in self.opt.frame_ids[1:]:
            #     pred = outputs[("color", frame_id, scale)]
            #     mssim_loss = 1 - self.ms_ssim_module(pred, target)
            #     mssim_loss *= 4 * warp_loss_weights[scale]
            #     loss += mssim_loss * global_loss_weights[2]
            #     losses[f"mssim_loss_{scale}"] = mssim_loss * global_loss_weights[2]


            total_loss += loss
            losses["loss/{}".format(scale)] = loss


        # conf_loss
        conf_loss = self.get_conf_loss(inputs, outputs)
        # w_loss
        w_pyramid = outputs["w_pyramid"]
        w_loss_pyramid = []
        w_loss_weights = [1/32, 1/32, 1/16, 1/16, 1/8, 1/8, 1/4, 1/4]
        for i, w in enumerate(w_pyramid):
            if i%2==0:
                j = i+1
            else:
                j = i-1
            w_loss_pyramid.append(
                w_loss_weights[i] * self.w_loss(w, w_pyramid[j], prop_diff_pyramid[i+1].detach(), prop_diff_pyramid[j+1].detach(), 
                                                ignore_mask==1,False, 1, 5)  # index for prop_diff_pyramid plus 1 since there is no confidence at 1st level
            )
        w_loss_vec = torch.cat(w_loss_pyramid, dim=0)

        total_loss /= self.num_scales
        
        w_loss = torch.mean(w_loss_vec)
        total_loss += w_loss * global_loss_weights[0]
        losses["loss/w_loss"] = w_loss * global_loss_weights[0]
    
        total_loss += conf_loss * global_loss_weights[1]
        losses["loss/conf_loss"] = conf_loss * global_loss_weights[1]  

        losses["loss"] = total_loss

        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def warp(self, x, disp, padding_mode='border'):
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

        if padding_mode == 'reflection':
            tmp = vgrid[:,:1,:,:] - disp
            tmp[tmp<0] = vgrid[:,:1,:,:][tmp<0] * (-1) - disp[tmp<0]
            vgrid[:,:1,:,:] = tmp
        else:
            # vgrid = Variable(grid)
            vgrid[:,:1,:,:] = vgrid[:,:1,:,:] - disp

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid, padding_mode=padding_mode)
        return output

    def w_loss(self, conf, conf2, diff, diff2, mask, high_conf_mask = False, C1=1, C2=1.5):
        """

        :param conf: aka omega
        :param diff: d^gt - d^
        :param C1:
        :param C2:
        :return: torch.Tensor: loss
        """
        conf = F.pad(conf, (0,diff.shape[-1]-conf.shape[-1],0,0))
        conf2 = F.pad(conf2, (0,diff.shape[-1]-conf2.shape[-1],0,0))
        closer_mask = (diff < C1) | high_conf_mask
        further_mask = diff > C2
        mask = mask * (closer_mask + further_mask)  # mask and
        
        punishment_mask = (conf > conf2) & (diff > diff2)
        
        closer_item = F.relu(1 - conf)
        further_item = F.relu(conf)
        # pdb.set_trace()
        loss = closer_item * closer_mask.float() + further_item * further_mask.float() + diff * punishment_mask.float()
        return loss[mask]  # 1-dim vector


    def get_opt(self):
            parser = argparse.ArgumentParser()
            opt = parser.parse_args([])

            opt.height = self.args.height
            opt.width = self.args.width
            opt.batch_size = self.args.batch_size
            opt.maxdisp = self.args.maxdisp
            opt.no_cuda = False
            opt.scales = list(range(0,12))
            opt.frame_ids = [0]
            opt.use_stereo = True

            opt.no_ssim = False  # TODO: check
            opt.device = self.args.device
            opt.loss_weights = self.args.loss_weights

            return opt
    

class cluster_loss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        with torch.no_grad():
            self.device = opt.device
            kernel_size = 5
            sigma = 1.5
            channel = len(opt.scales)
            kernel = 2*torch.tensor(self.gaussian_kernel(kernel_size, sigma)).unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1).to(self.device, dtype=torch.float)
            self.conv1 = nn.Conv2d(channel, channel, kernel_size, stride=1, padding=2, dilation=1, groups=channel, bias=None).to(self.device)
            self.conv1.weight.data = kernel
            self.pool1 = nn.MaxPool2d(5, 1, padding=2).to(self.device)
            self.opt = opt
            self.canny = CannyDetector(device=self.device)
            self.canny.to(self.opt.device)

    def forward(self, inputs, outputs):   
        # conf_loss
        conf_loss = 0.
        conf = outputs["conf"][...,:-inputs["shift_pixels"]]

        with torch.no_grad():
            # get edge image from Canny or edge map
            
            # image_canny = self.canny(gray)[5]+0.
            # image_canny = (inputs["edge"]<0.5).to(torch.float)
            image_canny = inputs["edge"]
            outputs["img_canny"] = image_canny

        # image_canny *= disp_grad_high.to(torch.float)
        nonzero_count = torch.count_nonzero(image_canny)
        conf_weight_tmp = ((torch.numel(image_canny)-nonzero_count) / (nonzero_count+1e-6))
        conf_entropy_loss = -conf_weight_tmp * image_canny * torch.log(conf+1e-6) - (1-image_canny) * torch.log(1-conf+1e-6)
        

        conf_loss += 0.1 * conf_entropy_loss.mean()

        return conf_loss
    


    def gaussian_kernel(self, kernel_size, sigma):
        x, y = np.meshgrid(np.arange(kernel_size), np.arange(kernel_size))
        x0 = (kernel_size - 1) / 2.0
        y0 = (kernel_size - 1) / 2.0
        d = ((x - x0) ** 2 + (y - y0) ** 2).ravel()
        k = np.exp(-d / (2 * sigma ** 2)).reshape([kernel_size,kernel_size]) / np.sum(np.exp(-d / (2 * sigma ** 2)))
        return k
    

    def pixelshuffle_invert(self, x: torch.Tensor, factor_hw):
        """pixelshuffle_invert

        Args:
            x (torch.Tensor): 
            factor_hw (Turple[int,int]): 

        """
        pH = factor_hw[0]
        pW = factor_hw[1]
        y = x
        B, iC, iH, iW = y.shape
        oC, oH, oW = iC*(pH*pW), iH//pH, iW//pW
        y = y.reshape(B, iC, oH, pH, oW, pW)
        y = y.permute(0, 1, 3, 5, 2, 4)     # B, iC, pH, pW, oH, oW
        y = y.reshape(B, oC, oH, oW)
        return y
    
    def rgb2gray(self, rgb: torch.Tensor):
        gray = 0.299*rgb[:,0,:,:] + 0.587*rgb[:,1,:,:] + 0.114*rgb[:,2,:,:]
        gray = gray.unsqueeze(1)
        return gray


    def fill_zeros_1d(self, disp):
        B,C,H,W = disp.shape
        disp_1d = disp.reshape(-1, disp.shape[3])

        x_index = torch.arange(0, W, device=disp.device).view(1, -1).repeat(H*B, 1)
        x = x_index.clone()
        x[disp_1d == 0] = -10000
        x_sorted, sorted_index = x.sort(dim = 1)
        disp_1d = torch.take_along_dim(disp_1d, sorted_index, dim = 1)

        disp_new_1d = interp1d(x_sorted, disp_1d, x_index)
        disp_filled = disp_new_1d.reshape(disp.shape)
        return disp_filled
    
    def fill_zeors_2d(self, disp):
        disp_filled_1d = self.fill_zeros_1d(disp)
        disp_permuted = disp_filled_1d.permute(0, 1, 3, 2)
        disp_filled_2d_permuted = self.fill_zeros_1d(disp_permuted)
        disp_filled_2d = disp_filled_2d_permuted.permute(0, 1, 3, 2)
        return disp_filled_2d
    
    
    def gradient_1order(self, x,h_x=None,w_x=None):
        if h_x is None and w_x is None:
            h_x = x.size()[2]
            w_x = x.size()[3]
        r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
        l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
        t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
        b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
        xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2), 0.5)
        return xgrad

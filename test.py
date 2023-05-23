from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models.HITNet import HITNet
from utils import *
from torch.utils.data import DataLoader
import gc
import json
from datetime import datetime
from utils.saver import Saver
import pdb
import losses.loss as total_loss
import imageio

cudnn.benchmark = True

class Tester:
    def __init__(self) -> None:
        def get_args():
            parser = argparse.ArgumentParser(description='HITNet')


            parser.add_argument('--maxdisp', type=int, default=192,
                                help='maximum disparity')
            parser.add_argument('--fea_c', type=list,
                                default=[32, 24, 24, 16, 16], help='feature extraction channels')

            parser.add_argument('--dataset', default="kitti",
                                help='dataset name', choices=__datasets__.keys())
            parser.add_argument('--datapath', help='data path', default="/home/hey/SIA/cv/dataset/kitti")

            parser.add_argument('--testlist',
                                help='testing list', default="filenames/kt2015_val_first20.txt")
                                # help='testing list', default="filenames/kt15_testing.txt")


            parser.add_argument('--lr', type=float, default=0.001,
                                help='base learning rate')
            parser.add_argument('--batch_size', type=int, default=5,
                                help='training batch size')
            parser.add_argument('--test_batch_size', type=int,
                                default=8, help='testing batch size')
            parser.add_argument('--epochs', type=int, default=800,
                                help='number of epochs to train')

            parser.add_argument('--logdir', default="./logs",
                                help='the directory to save logs and checkpoints')
            parser.add_argument(
                '--loadckpt', help='load the weights from a specific checkpoint')
            parser.add_argument('--resume', type=str, 
                help='read the model')
            parser.add_argument('--seed', type=int, default=1,
                                metavar='S', help='random seed (default: 1)')

            parser.add_argument('--summary_freq', type=int, default=20,
                                help='the frequency of saving summary')
            parser.add_argument('--save_freq', type=int, default=10,
                                help='the frequency of saving checkpoint')

            parser.add_argument('--save_disp_to_file', action="store_true",
                                help='save estimated disp maps to files')
            return parser.parse_args()

        args = get_args()
        args.device = 'cuda'
        args.loss_weights = [0.01, 0.01, 1, 0, 1, 1]
        self.args = args
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        os.makedirs(args.logdir, exist_ok=True)

        # create summary self.logger
        self.saver = Saver(args)
        print("creating new summary file")
        self.logger = SummaryWriter(self.saver.experiment_dir)

        self.logfilename = self.saver.experiment_dir + '/log.txt'

        with open(self.logfilename, 'a') as log:  # wrt running information to log
            log.write('\n\n\n\n')
            log.write('-------------------NEW RUN-------------------\n')
            log.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            log.write('\n')
            json.dump(args.__dict__, log, indent=2)
            log.write('\n')

        # dataset, dataloader
        self.StereoDataset = __datasets__[args.dataset]

        self.test_dataset = self.StereoDataset(args.datapath, args.testlist, False)

        self.TestImgLoader = DataLoader(self.test_dataset, args.test_batch_size, shuffle=False, 
                                        num_workers=8, drop_last=False, pin_memory=True)

        # self.model, self.optimizer
        self.model = HITNet(args)
        self.model = nn.DataParallel(self.model)
        self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999))

        # load parameters
        self.start_epoch = 0
        if args.resume:
            print("loading the lastest self.model in logdir: {}".format(args.resume))
            self.state_dict = torch.load(args.resume)
            self.model.load_state_dict(self.state_dict['self.model'])
            self.optimizer.load_state_dict(self.state_dict['self.optimizer'])
            self.start_epoch = self.state_dict['epoch'] + 1
        elif args.loadckpt:
            # load the checkpoint file specified by args.loadckpt
            print("loading self.model {}".format(args.loadckpt))
            self.state_dict = torch.load(args.loadckpt)
            self.model.load_state_dict(self.state_dict['self.model'])
        print("start at epoch {}".format(self.start_epoch))


    def run(self):
        args = self.args
        min_EPE = args.maxdisp
        min_D1 = 1
        min_Thres3 = 1
        
        epoch_idx = self.start_epoch
        avg_test_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(self.TestImgLoader):
            # if batch_idx == 2:
            #     break
            global_step = len(self.TestImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = (global_step) % args.summary_freq== 0
            scalar_outputs, image_outputs = self.test_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(self.logger, 'test', scalar_outputs, global_step)
                save_images(self.logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                batch_idx,
                                                                                len(self.TestImgLoader), 
                                                                                time.time() - start_time))
            with open(self.logfilename, 'a') as log:
                log.write('Epoch {}/{}, Iter {}/{}, time = {:.3f}\n'.format(epoch_idx, args.epochs,
                                                                                            batch_idx,
                                                                                            len(self.TestImgLoader),
                                                                                            time.time() - start_time))
        avg_test_scalars = avg_test_scalars.mean()

        print("avg_test_scalars", avg_test_scalars)
        with open(self.logfilename, 'a') as log:
            js = json.dumps(avg_test_scalars)
            log.write(js)
            log.write('\n')
        gc.collect()

    # test one sample
    @make_nograd_func
    def test_sample(self, sample, compute_metrics=True):
        args = self.args
        args.height = sample['left'].shape[2]
        args.width = sample['left'].shape[3]
        self.model.eval()

        if ('disparity' not in sample): 
            disp_gt = None
            mask = True
        else:
            disp_gt = sample['disparity']
            disp_gt = disp_gt.to(args.device)
            mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
        imgL, imgR = sample['left'], sample['right']
        
        imgL = imgL.cuda()
        imgR = imgR.cuda()
        if disp_gt is not None:
            disp_gt = disp_gt.cuda().unsqueeze(1)


        outputs = self.model(imgL, imgR)
        prop_disp_pyramid = outputs['prop_disp_pyramid']

        scalar_outputs = {}
        img_conf = imgL.clone().permute(0, 2, 3, 1)
        img_conf[outputs["conf"].squeeze(1)>0.5, :] = torch.tensor([193.0/256, 44.0/256, 31.0/256], device = args.device)
        img_conf = img_conf.permute(0, 3, 1, 2)
        img_conf_1 = (outputs["conf"].squeeze(1)>0.5).to(torch.float)
        image_outputs = {"disp_est": prop_disp_pyramid, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR,
                         "img_conf": img_conf, "img_conf_1": img_conf_1}
        if disp_gt is not None:
            scalar_outputs["D1"] = [D1_metric(disp_est.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1)) for disp_est in prop_disp_pyramid]
            scalar_outputs["EPE"] = [EPE_metric(disp_est.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1)) for disp_est in prop_disp_pyramid]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1), 1.0) for disp_est in prop_disp_pyramid]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1), 2.0) for disp_est in prop_disp_pyramid]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1), 3.0) for disp_est in prop_disp_pyramid]
        
        if args.save_disp_to_file:
            self.save_disp_tofile(prop_disp_pyramid[-1], sample)

        if compute_metrics & (disp_gt is not None):
            image_outputs["errormap"] = [disp_error_image_func.apply(disp_est.squeeze(1), disp_gt.squeeze(1)) for disp_est in prop_disp_pyramid]

        return tensor2float(scalar_outputs), image_outputs
    
    @make_nograd_func
    def save_disp_tofile(self, disp, sample):
        filename = sample["left_filename"][0].split('/')[-1]
        path = self.args.datapath
        full_path = os.path.join(path,'kitti_2015/testing/disp_0',filename)
        top = sample["top_pad"]
        right = sample["right_pad"]
        disp_np=np.array(disp[:,:,top:,:-right].squeeze().detach().cpu()*256).astype(np.uint16)
        imageio.imsave(full_path,disp_np)

if __name__ == '__main__':
    tester = Tester()
    tester.run()

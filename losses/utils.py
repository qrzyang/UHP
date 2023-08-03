from __future__ import absolute_import, division, print_function
import os, sys
import hashlib
import zipfile
from six.moves import urllib
import torch, torch.nn as nn, numpy as np
sys.path.append(os.path.dirname(__file__))
from torchinterp1d import interp1d
import cv2
from scipy.signal import gaussian
import math


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def download_model_if_doesnt_exist(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
             "a964b8356e08a02d009609d9e3928f7c"),
        "stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
             "3dfb76bcff0786e4ec07ac00f658dd07"),
        "mono+stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
             "c024d69012485ed05d7eaa9617a96b81"),
        "mono_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
             "9c2f071e35027c895a4728358ffc913a"),
        "stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
             "41ec2de112905f85541ac33a854742d1"),
        "mono+stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
             "46c3b824f541d143a45c37df65fbab0a"),
        "mono_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
             "0ab0766efdfeea89a0d9ea8ba90e1e63"),
        "stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
             "afc2f2126d70cf3fdf26b550898b501a"),
        "mono+stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
             "cdc5fc9b23513c07d5b19235d9ef08f7"),
        }

    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))


def fill_zeros_1d(disp):
    B,C,H,W = disp.shape
    disp_1d = disp.reshape(-1, disp.shape[3])
    # x_sorted, x_index = disp_1d.sort(dim=1)
    # x_index[x_sorted == 0] = -1000
    # x_new_index = torch.arange(0, W, device=disp.device).view(1, -1).repeat(H*B, 1)

    # disp_new_1d = interp1d(x_index, x_sorted, x_new_index)
    # disp_filled = disp_new_1d.reshape(disp.shape)

    x_index = torch.arange(0, W, device=disp.device).view(1, -1).repeat(H*B, 1)
    x = x_index.clone()
    x[disp_1d == 0] = -10000
    x_sorted, sorted_index = x.sort(dim = 1)
    disp_1d = torch.take_along_dim(disp_1d, sorted_index, dim = 1)

    # x_index[disp_1d!=0] = -10
    disp_new_1d = interp1d(x_sorted, disp_1d, x_index)
    disp_filled = disp_new_1d.reshape(disp.shape)
    return disp_filled

def fill_zeros_2d(disp):
    disp_filled_1d = fill_zeros_1d(disp)
    disp_permuted = disp_filled_1d.permute(0, 1, 3, 2)
    disp_filled_2d_permuted = fill_zeros_1d(disp_permuted)
    disp_filled_2d = disp_filled_2d_permuted.permute(0, 1, 3, 2)
    return disp_filled_2d


def get_state_dict(filter_size=5, std=1.0, map_func=lambda x:x):
    generated_filters = gaussian(filter_size, std=std).reshape([1, filter_size
                                                   ]).astype(np.float32)

    gaussian_filter_horizontal = generated_filters[None, None, ...]

    gaussian_filter_vertical = generated_filters.T[None, None, ...]

    sobel_filter_horizontal = np.array([[[
        [1., 0., -1.], 
        [2., 0., -2.],
        [1., 0., -1.]]]], 
        dtype='float32'
    )

    sobel_filter_vertical = np.array([[[
        [1., 2., 1.], 
        [0., 0., 0.], 
        [-1., -2., -1.]]]], 
        dtype='float32'
    )

    directional_filter = np.array(
        [[[[ 0.,  0.,  0.],
          [ 0.,  1., -1.],
          [ 0.,  0.,  0.]]],


        [[[ 0.,  0.,  0.],
          [ 0.,  1.,  0.],
          [ 0.,  0., -1.]]],


        [[[ 0.,  0.,  0.],
          [ 0.,  1.,  0.],
          [ 0., -1.,  0.]]],


        [[[ 0.,  0.,  0.],
          [ 0.,  1.,  0.],
          [-1.,  0.,  0.]]],


        [[[ 0.,  0.,  0.],
          [-1.,  1.,  0.],
          [ 0.,  0.,  0.]]],


        [[[-1.,  0.,  0.],
          [ 0.,  1.,  0.],
          [ 0.,  0.,  0.]]],


        [[[ 0., -1.,  0.],
          [ 0.,  1.,  0.],
          [ 0.,  0.,  0.]]],


        [[[ 0.,  0., -1.],
          [ 0.,  1.,  0.],
          [ 0.,  0.,  0.]]]], 
        dtype=np.float32
    )

    connect_filter = np.array([[[
        [1., 1., 1.], 
        [1., 0., 1.], 
        [1., 1., 1.]]]],
        dtype=np.float32
    )

    return {
        'gaussian_filter_horizontal.weight': map_func(gaussian_filter_horizontal),
        'gaussian_filter_vertical.weight': map_func(gaussian_filter_vertical),
        'sobel_filter_horizontal.weight': map_func(sobel_filter_horizontal),
        'sobel_filter_vertical.weight': map_func(sobel_filter_vertical),
        'directional_filter.weight': map_func(directional_filter),
        'connect_filter.weight': map_func(connect_filter)
    }


# https://github.com/jm12138/CannyDetector
class CannyDetector(nn.Module):
    def __init__(self, filter_size=5, std=1.0, device='cpu'):
        super(CannyDetector, self).__init__()
        # 配置运行设备
        self.device = device

        # 高斯滤波器
        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2), bias=False)
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0), bias=False)

        # Sobel 滤波器
        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        # 定向滤波器
        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, bias=False)

        # 连通滤波器
        self.connect_filter = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        # 初始化参数
        params = get_state_dict(filter_size=filter_size, std=std, map_func=lambda x:torch.from_numpy(x).to(self.device))
        self.load_state_dict(params)

    @torch.no_grad()
    def forward(self, img, threshold1=2.5, threshold2=5):
        # 拆分图像通道
        img_r = img[:,0:1] # red channel
        img_g = img[:,1:2] # green channel
        img_b = img[:,2:3] # blue channel

        # Step1: 应用高斯滤波进行模糊降噪
        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        # Step2: 用 Sobel 算子求图像的强度梯度
        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # Step2: 确定边缘梯度和方向
        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2)
        grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2)
        grad_orientation = (torch.atan2(grad_y_r+grad_y_g+grad_y_b, grad_x_r+grad_x_g+grad_x_b) * (180.0/math.pi))
        grad_orientation += 180.0
        grad_orientation =  torch.round(grad_orientation / 45.0) * 45.0

        # Step3: 非最大抑制，边缘细化
        all_filtered = self.directional_filter(grad_mag)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        batch, _, height, width = inidices_positive.shape
        pixel_count = height * width * batch
        pixel_range = torch.Tensor([range(pixel_count)]).to(self.device)

        indices = (inidices_positive.reshape((-1, )) * pixel_count + pixel_range).squeeze()
        channel_select_filtered_positive = all_filtered.reshape((-1, ))[indices.long()].reshape((batch, 1, height, width))

        indices = (inidices_negative.reshape((-1, )) * pixel_count + pixel_range).squeeze()
        channel_select_filtered_negative = all_filtered.reshape((-1, ))[indices.long()].reshape((batch, 1, height, width))

        channel_select_filtered = torch.stack([channel_select_filtered_positive, channel_select_filtered_negative])

        is_max = channel_select_filtered.min(dim=0)[0] > 0.0

        thin_edges = grad_mag.clone()
        thin_edges[is_max==0] = 0.0

        # Step4: 双阈值
        low_threshold = min(threshold1, threshold2)
        high_threshold = max(threshold1, threshold2)
        thresholded = thin_edges.clone()
        lower = thin_edges<low_threshold
        thresholded[lower] = 0.0
        higher = thin_edges>high_threshold
        thresholded[higher] = 1.0
        connect_map = self.connect_filter(higher.float())
        middle = torch.logical_and(thin_edges>=low_threshold, thin_edges<=high_threshold)
        thresholded[middle] = 0.0
        connect_map[torch.logical_not(middle)] = 0
        thresholded[connect_map>0] = 1.0
        thresholded[..., 0, :] = 0.0
        thresholded[..., -1, :] = 0.0
        thresholded[..., :, 0] = 0.0
        thresholded[..., :, -1] = 0.0
        thresholded = (thresholded>0.0).float()

        return thresholded

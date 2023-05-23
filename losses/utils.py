from __future__ import absolute_import, division, print_function
import os, sys
import hashlib
import zipfile
from six.moves import urllib
import torch, torch.nn as nn, numpy as np
sys.path.append(os.path.dirname(__file__))
from torchinterp1d import interp1d
import cv2



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


class CannyFilter(nn.Module):
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 device = 'cpu'):
        super(CannyFilter, self).__init__()
        # device
        self.device = device
        
        # gaussian
        with torch.no_grad():
            gaussian_2D = self.get_gaussian_kernel(k_gaussian, mu, sigma)
            self.gaussian_filter = nn.Conv2d(in_channels=1,
                                            out_channels=1,
                                            kernel_size=k_gaussian,
                                            padding=k_gaussian // 2,
                                            bias=False)
            self.gaussian_filter.weight[:] = torch.from_numpy(gaussian_2D)

            # sobel

            sobel_2D = self.get_sobel_kernel(k_sobel)
            self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                            out_channels=1,
                                            kernel_size=k_sobel,
                                            padding=k_sobel // 2,
                                            bias=False)
            self.sobel_filter_x.weight[:] = torch.from_numpy(sobel_2D)


            self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                            out_channels=1,
                                            kernel_size=k_sobel,
                                            padding=k_sobel // 2,
                                            bias=False)
            self.sobel_filter_y.weight[:] = torch.from_numpy(sobel_2D.T)


            # thin

            thin_kernels = self.get_thin_kernels()
            directional_kernels = np.stack(thin_kernels)

            self.directional_filter = nn.Conv2d(in_channels=1,
                                                out_channels=8,
                                                kernel_size=thin_kernels[0].shape,
                                                padding=thin_kernels[0].shape[-1] // 2,
                                                bias=False)
            self.directional_filter.weight[:, 0] = torch.from_numpy(directional_kernels)

            # hysteresis

            hysteresis = np.ones((3, 3)) + 0.25
            self.hysteresis = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=3,
                                        padding=1,
                                        bias=False)
            self.hysteresis.weight[:] = torch.from_numpy(hysteresis)



    def get_gaussian_kernel(self, k=3, mu=0, sigma=2, normalize=True):
        # compute 1 dimension gaussian
        gaussian_1D = np.linspace(-1, 1, k)
        # compute a grid distance from center
        x, y = np.meshgrid(gaussian_1D, gaussian_1D)
        distance = (x ** 2 + y ** 2) ** 0.5

        # compute the 2 dimension gaussian
        gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
        gaussian_2D = gaussian_2D / (2 * np.pi *sigma **2)

        # normalize part (mathematically)
        if normalize:
            gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
        return gaussian_2D

    
    def get_sobel_kernel(self, k=3):
        # get range
        range = np.linspace(-(k // 2), k // 2, k)
        # compute a grid the numerator and the axis-distances
        x, y = np.meshgrid(range, range)
        sobel_2D_numerator = x
        sobel_2D_denominator = (x ** 2 + y ** 2)
        sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
        sobel_2D = sobel_2D_numerator / sobel_2D_denominator
        return sobel_2D

    def get_thin_kernels(self, start=0, end=360, step=45):
            k_thin = 3  # actual size of the directional kernel
            # increase for a while to avoid interpolation when rotating
            k_increased = k_thin + 2

            # get 0° angle directional kernel
            thin_kernel_0 = np.zeros((k_increased, k_increased))
            thin_kernel_0[k_increased // 2, k_increased // 2] = 1
            thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

            # rotate the 0° angle directional kernel to get the other ones
            thin_kernels = []
            for angle in range(start, end, step):
                (h, w) = thin_kernel_0.shape
                # get the center to not rotate around the (0, 0) coord point
                center = (w // 2, h // 2)
                # apply rotation
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
                kernel_angle_increased = cv2.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)

                # get the k=3 kerne
                kernel_angle = kernel_angle_increased[1:-1, 1:-1]
                is_diag = (abs(kernel_angle) == 1)      # because of the interpolation
                kernel_angle = kernel_angle * is_diag   # because of the interpolation
                thin_kernels.append(kernel_angle)
            return thin_kernels

    def forward(self, img, low_threshold=0.05, high_threshold=0.25, hysteresis=True):
        with torch.no_grad():
            # set the setps tensors
            B, C, H, W = img.shape
            blurred = torch.zeros((B, C, H, W)).to(self.device)
            grad_x = torch.zeros((B, 1, H, W)).to(self.device)
            grad_y = torch.zeros((B, 1, H, W)).to(self.device)
            grad_magnitude = torch.zeros((B, 1, H, W)).to(self.device)
            grad_orientation = torch.zeros((B, 1, H, W)).to(self.device)

            # gaussian

            for c in range(C):
                blurred[:, c:c+1] = self.gaussian_filter(img[:, c:c+1])

                grad_x = grad_x + self.sobel_filter_x(blurred[:, c:c+1])
                grad_y = grad_y + self.sobel_filter_y(blurred[:, c:c+1])

            # thick edges

            grad_x, grad_y = grad_x / C, grad_y / C
            grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5
            grad_orientation = torch.atan(grad_y / grad_x)
            grad_orientation = grad_orientation * (360 / np.pi) + 180 # convert to degree
            grad_orientation = torch.round(grad_orientation / 45) * 45  # keep a split by 45

            # thin edges

            directional = self.directional_filter(grad_magnitude)
            # get indices of positive and negative directions
            positive_idx = (grad_orientation / 45) % 8
            negative_idx = ((grad_orientation / 45) + 4) % 8
            thin_edges = grad_magnitude.clone()
            # non maximum suppression direction by direction
            for pos_i in range(4):
                neg_i = pos_i + 4
                # get the oriented grad for the angle
                is_oriented_i = (positive_idx == pos_i) * 1
                is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
                pos_directional = directional[:, pos_i]
                neg_directional = directional[:, neg_i]
                selected_direction = torch.stack([pos_directional, neg_directional])

                # get the local maximum pixels for the angle
                is_max = selected_direction.min(dim=0)[0] > 0.0
                is_max = torch.unsqueeze(is_max, dim=1)

                # apply non maximum suppression
                to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
                thin_edges[to_remove] = 0.0

            # thresholds

            if low_threshold is not None:
                low = thin_edges > low_threshold

                if high_threshold is not None:
                    high = thin_edges > high_threshold
                    # get black/gray/white only
                    thin_edges = low * 0.5 + high * 0.5

                    if hysteresis:
                        # get weaks and check if they are high or not
                        weak = (thin_edges == 0.5) * 1
                        weak_is_high = (self.hysteresis(thin_edges) > 1) * weak
                        thin_edges = high * 1 + weak_is_high * 1
                else:
                    thin_edges = low * 1


        return blurred, grad_x, grad_y, grad_magnitude, grad_orientation, thin_edges
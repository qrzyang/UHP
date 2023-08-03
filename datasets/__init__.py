# from .kitti_dataset import KITTIDataset
from .kitti_y import KITTIDataset, KITTI_Raw
from .kitti_y import MyCollator
__datasets__ = {
    "kitti": KITTIDataset, 
    "kitti_raw": KITTI_Raw
}

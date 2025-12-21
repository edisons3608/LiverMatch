import sys
[sys.path.append(i) for i in ['.', '..']]
import numpy as np
import torch
import random
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

HMN_intrin = np.array([443, 256, 443, 250])
cam_intrin = np.array([443, 256, 443, 250])
from lib.visualization import viz_flow_mayavi, viz_coarse_nn_correspondence_mayavi, compare_pcd
from lib.util import to_o3d_pcd, to_tsfm, get_correspondences_n
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import pyvista as pv

class liverPhantom(Dataset):

    def __init__(self, config, mode):
        super(liverPhantom, self).__init__()
        self.root_path = config.root_path
        self.mode = mode
        if mode =="train":
            self.file_list = np.loadtxt(config.train_list, dtype=str).tolist()
        else:
            self.test_list = np.loadtxt(config.test_list, dtype=str).tolist()
        # augmentation parameters
        self.max_noise = config.max_noise # random noise
        self.overlap_radius = config.overlap_radius # as noise is added, searching new corr
        self.max_vis = config.max_vis
        self.min_vis = config.min_vis
        self.config = config



    def __len__(self):
        if self.mode == "train":
            return len(self.file_list)
        else:
            return len(self.test_list)

    def crop(self, points, p_keep, rand_xyz=None):
        if rand_xyz is None:
            rand_xyz = self.uniform_2_sphere()
        centroid = np.mean(points[:, :3], axis=0)
        points_centered = points[:, :3] - centroid

        dist_from_plane = np.dot(points_centered, rand_xyz)
        if p_keep == 0.5:
            mask = dist_from_plane > 0
        else:
            mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)

        return points[mask, :], mask, rand_xyz

    def uniform_2_sphere(self, num: int = None):
        """Uniform sampling on a 2-sphere
        Source: https://gist.github.com/andrewbolster/10274979
        Args:
            num: Number of vectors to sample (or None if single)
        Returns:
            Random Vector (np.ndarray) of size (num, 3) with norm 1.
            If num is None returned value will have size (3,)

        """
        if num is not None:
            phi = np.random.uniform(0.0, 2 * np.pi, num)
            cos_theta = np.random.uniform(-1.0, 1.0, num)
        else:
            phi = np.random.uniform(0.0, 2 * np.pi)
            cos_theta = np.random.uniform(-1.0, 1.0)

        theta = np.arccos(cos_theta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        return np.stack((x, y, z), axis=-1)

    def rand_rot(self, pcd, euler_ab=None):
        if euler_ab is None:
            euler_ab = np.random.rand(3) * np.pi * 2
        rot = Rotation.from_euler('zyx', euler_ab).as_matrix()
        pcd = (np.matmul(rot, pcd.T)).T
        return pcd

    def center(self, points):
        centroid = np.mean(points[:, :3], axis=0)
        points_centered = points[:, :3] - centroid
        return points_centered

    def get_input_train(self, index, vis=False, m=100.0, p=None):
        #group = self.file_list[str(index)]


        pair_file = random.sample( self.file_list, 2)

        src_file = self.root_path + pair_file[0]
        tgt_file = self.root_path + pair_file[1]

        src_vs = pv.read(src_file).points
        tgt_vs_full = pv.read(tgt_file).points

        # random crop
        if p is None:
            p = self.min_vis + (self.max_vis - self.min_vis) * np.random.rand(1)[0]
            if p > 1:
                p = 1.0

        tgt_pcd, mask, rand_xyz = self.crop(tgt_vs_full, p)

        # random noise
        sigma = np.random.rand(1)[0] * self.max_noise
        tgt_pcd += (np.random.rand(tgt_pcd.shape[0], 3) - 0.5) * sigma

        # np.random.normal(0, sigma, tgt_pcd.shape)

        # search corr
        correspondences = get_correspondences_n(to_o3d_pcd(src_vs), to_o3d_pcd(tgt_pcd),
                                                self.overlap_radius)

        # random rotation src and tgt
        src_pcd = self.rand_rot(src_vs)
        tgt_pcd = self.rand_rot(tgt_pcd)

        # center and scale
        # move to center, so actually it does not care about the trainsition

        src_pcd = self.center(src_pcd)

        m = np.max(np.sqrt(np.sum(src_pcd ** 2, axis=1)))

        src_pcd = src_pcd / m
        tgt_pcd = self.center(tgt_pcd) / m

        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)
        rot = np.zeros([3, 3])
        trans = np.zeros([3, 1])

        if vis:
            print("num corr", len(correspondences))
            print("vis ratio", p)
            scale_factor = 0.013
            viz_coarse_nn_correspondence_mayavi(src_pcd, tgt_pcd + 1, correspondences.T, f_src_pcd=None,
                                                f_tgt_pcd=None,
                                                scale_factor=scale_factor)

        return src_pcd, tgt_pcd, src_feats, tgt_feats, \
            rot, trans, correspondences, src_vs, tgt_pcd, torch.ones(1)

    def __getitem__(self, index, vis=False, sigma=None):

        if self.mode == "train":
            return self.get_input_train(index, vis=vis)

if __name__ == '__main__':

    from easydict import EasyDict as edict
    from dataloader import calibrate_neighbors, collate_fn_descriptor
    from configs.models import architectures
    from lib.util import load_config

    config_path = "/home/yzx/yzx/Deformable_Registration/LiverMatch/configs/liver_phantom.yaml"
    config = load_config(config_path)
    config = edict(config)

    dataset = liverPhantom(config, "train")
    # data.get_input_train(0, vis=True, p=0.18)
    data = dataset.__getitem__(10, vis=True)

    config.architecture = architectures[config.model_name]

    neighborhood_limits = calibrate_neighbors(dataset, config, collate_fn_descriptor)
    print(neighborhood_limits)



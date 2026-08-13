import sys
[sys.path.append(i) for i in ['.', '..']]
import random

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class talusDataset(Dataset):
    """Same-bone partial-view rigid registration (source = full bone, target = a
    cropped/noised/rotated view of the IDENTICAL bone). Correspondences are exact
    (derived from the crop mask, not a proximity guess) since source and target
    are literally the same point cloud before cropping -- same style as
    `livermatch`'s full-vs-partial-target setup, but for a single rigid bone
    instead of a deforming liver.
    """

    def __init__(self, config, mode):
        super(talusDataset, self).__init__()
        self.root_path = config.root_path
        self.mode = mode
        if mode == "train":
            self.file_list = np.loadtxt(config.train_list, dtype=str).tolist()
        else:
            self.file_list = np.loadtxt(config.val_list, dtype=str).tolist()

        self.max_noise = config.max_noise
        self.overlap_radius = config.overlap_radius
        self.max_vis = config.max_vis
        self.min_vis = config.min_vis
        self.config = config

    def __len__(self):
        return len(self.file_list)

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
        return points[:, :3] - centroid

    def get_input_train(self, index, vis=False, p=None):
        src_vs = np.load(self.root_path + random.choice(self.file_list))
        tgt_vs_full = src_vs  # same bone: target is a partial/noised/rotated view of the identical cloud

        if p is None:
            p = self.min_vis + (self.max_vis - self.min_vis) * np.random.rand(1)[0]
            if p > 1:
                p = 1.0

        tgt_pcd, mask, rand_xyz = self.crop(tgt_vs_full, p)

        sigma = np.random.rand(1)[0] * self.max_noise
        tgt_pcd = tgt_pcd + (np.random.rand(tgt_pcd.shape[0], 3) - 0.5) * sigma

        # exact correspondence: point i in the full source survives the crop at
        # position j in tgt_pcd iff mask[i] is True and j is its rank among kept points
        src_idx = np.nonzero(mask)[0]
        tgt_idx = np.arange(len(src_idx))
        correspondences = torch.from_numpy(np.stack([src_idx, tgt_idx], axis=1)).long()

        src_pcd = self.rand_rot(src_vs)
        tgt_pcd = self.rand_rot(tgt_pcd)

        # keep the cache's pre-baked target_diag scale (KPConv radii are absolute,
        # calibrated to the pretrained network); do not rescale to a unit sphere here.
        src_pcd = self.center(src_pcd)
        tgt_pcd = self.center(tgt_pcd)

        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)
        rot = np.zeros([3, 3])
        trans = np.zeros([3, 1])

        if correspondences.size(0) < 20:
            return self.get_input_train(np.random.choice(len(self.file_list), 1)[0], vis=vis)

        return src_pcd.astype(np.float32), tgt_pcd.astype(np.float32), src_feats, tgt_feats, \
            rot.astype(np.float32), trans.astype(np.float32), correspondences, src_vs, tgt_pcd.astype(np.float32), torch.ones(1)

    def __getitem__(self, index, vis=False):
        return self.get_input_train(index, vis=vis)


if __name__ == '__main__':
    from easydict import EasyDict as edict
    from lib.util import load_config

    config_path = "configs/talus.yaml"
    config = load_config(config_path)
    config = edict(config)

    dataset = talusDataset(config, "train")
    data = dataset.__getitem__(0)
    print("src", data[0].shape, "tgt", data[1].shape, "n_corr", data[6].shape)

"""Compare the original pretrained LiverMatch checkpoint against the talus fine-tune(s)
on two scenarios:
  1. Cross-subject (same as talus_demo.py): two different subjects' full bones.
  2. Same-bone partial view: one subject's bone, cropped/noised, under a known rigid
     transform -- the actual target use case (registering the same bone to itself).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from easydict import EasyDict as edict
from scipy.spatial.transform import Rotation

from lib.util import load_config
from configs.models import architectures
from models.framework import KPFCNN
from datasets.dataloader import get_dataloader, collate_fn_descriptor

from talus_demo import stl_to_pcd, eva_regist, rot_trans_error, chamfer_like, PairDemo

import warnings
warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TALUS_DIR = r"C:\Users\esun3\Documents\talus_small"

config_path = os.path.join(REPO_ROOT, "configs", "liver.yaml")
config = load_config(config_path)
config = edict(config)
config.architecture = architectures[config.model_name]
config.device = torch.device('cuda:0')

checkpoints = {
    "pretrained (original)":         os.path.join(REPO_ROOT, "snapshot", "liver_3D_1_one_transformer",
                                                  "checkpoints", "model_best_loss.pth"),
    "talus fine-tuned (9ep self-pair)":  os.path.join(REPO_ROOT, "snapshot", "talus_finetune_selfpair",
                                                  "checkpoints", "model_best_loss.pth"),
    "talus fine-tuned (30ep self-pair)": os.path.join(REPO_ROOT, "snapshot", "talus_finetune_selfpair_30ep",
                                                  "checkpoints", "model_best_loss.pth"),
}


def uniform_2_sphere():
    phi = np.random.uniform(0.0, 2 * np.pi)
    cos_theta = np.random.uniform(-1.0, 1.0)
    theta = np.arccos(cos_theta)
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def crop(points, p_keep):
    rand_xyz = uniform_2_sphere()
    centroid = points.mean(0)
    dist = np.dot(points - centroid, rand_xyz)
    mask = dist > np.percentile(dist, (1.0 - p_keep) * 100)
    return points[mask]


def build_cross_subject_scenario():
    src_stl = os.path.join(TALUS_DIR, "200001-xx-f-055xxxxxx-tal-l-c-d-s%.stl")
    tgt_stl = os.path.join(TALUS_DIR, "200002-xx-m-068xxxxxx-tal-l-c-d-s%.stl")
    np.random.seed(0)
    src_pcd = stl_to_pcd(src_stl, n_points=8000, seed=0)
    tgt_pcd_raw = stl_to_pcd(tgt_stl, n_points=8000, seed=1)
    euler_gt = np.random.uniform(-np.pi / 3, np.pi / 3, size=3)
    rot_gt = Rotation.from_euler('zyx', euler_gt).as_matrix().astype(np.float32)
    trans_gt = (np.random.rand(3, 1) * 0.6 - 0.3).astype(np.float32)
    tgt_pcd = (np.matmul(rot_gt, tgt_pcd_raw.T) + trans_gt).T
    desc = f"Cross-subject: {os.path.basename(src_stl)} -> {os.path.basename(tgt_stl)}"
    return desc, src_pcd, tgt_pcd, rot_gt, trans_gt


def build_same_bone_scenario():
    src_stl = os.path.join(TALUS_DIR, "200001-xx-f-055xxxxxx-tal-l-c-d-s%.stl")
    np.random.seed(1)
    full_pcd = stl_to_pcd(src_stl, n_points=8000, seed=0)
    src_pcd = full_pcd
    tgt_partial = crop(full_pcd, p_keep=0.6)
    tgt_partial = tgt_partial + (np.random.rand(*tgt_partial.shape) - 0.5) * 0.02
    euler_gt = np.random.uniform(-np.pi / 3, np.pi / 3, size=3)
    rot_gt = Rotation.from_euler('zyx', euler_gt).as_matrix().astype(np.float32)
    trans_gt = (np.random.rand(3, 1) * 0.6 - 0.3).astype(np.float32)
    tgt_pcd = (np.matmul(rot_gt, tgt_partial.T) + trans_gt).T
    desc = f"Same-bone partial view: {os.path.basename(src_stl)} (60% visible, +noise)"
    return desc, src_pcd, tgt_pcd, rot_gt, trans_gt


scenarios = {
    "cross-subject (talus_demo.py)": build_cross_subject_scenario(),
    "same-bone partial view":        build_same_bone_scenario(),
}

neighborhood_limits = [19, 23, 29, 34]

for scenario_name, (desc, src_pcd, tgt_pcd, rot_gt, trans_gt) in scenarios.items():
    before_dist = chamfer_like(src_pcd, tgt_pcd)
    demo_set = PairDemo(config, src_pcd, tgt_pcd)
    list_data = demo_set.__getitem__(0)
    inputs = collate_fn_descriptor([list_data], config, neighborhood_limits)

    print(f"\n########## {scenario_name} ##########")
    print(desc)
    print(f"Mean NN dist src->tgt before alignment: {before_dist:.4f}")

    for name, ckpt_path in checkpoints.items():
        model = KPFCNN(config).to(config.device).eval()
        state = torch.load(ckpt_path, map_location=config.device)
        model.load_state_dict(state['state_dict'])

        with torch.no_grad():
            dev_inputs = {}
            for k, v in inputs.items():
                dev_inputs[k] = [item.to(config.device) for item in v] if isinstance(v, list) else v.to(config.device)
            data = model(dev_inputs)

        match_pred = data['match_pred'].detach().cpu()[:, 1:]
        scores_vis = data['scores_vis'].detach().cpu()

        th_score = 0.9
        vis_ok_ids = torch.nonzero(scores_vis > th_score, as_tuple=True)[0]
        keep = torch.isin(match_pred[:, 0], vis_ok_ids)
        match_pred_scores = match_pred[keep]

        n_raw = len(match_pred)
        n_high_conf = len(match_pred_scores)
        if n_high_conf < 4:
            match_pred_scores = match_pred

        tsfm_pred = eva_regist(src_pcd, tgt_pcd, match_pred_scores, distance_threshold=0.15, ransac_n=4)
        rot_err_deg, trans_err = rot_trans_error(tsfm_pred, rot_gt, trans_gt)

        src_aligned = (np.matmul(tsfm_pred[:3, :3], src_pcd.T) + tsfm_pred[:3, 3:]).T
        after_dist = chamfer_like(src_aligned, tgt_pcd)

        print(f"  === {name} ===")
        print(f"    raw correspondences: {n_raw}, high-confidence (>{th_score}): {n_high_conf}")
        print(f"    rot_error_deg: {rot_err_deg:.3f}   trans_error: {trans_err:.4f}")
        print(f"    mean NN dist after alignment: {after_dist:.4f}")


"""Compare the original pretrained LiverMatch checkpoint against the talus fine-tune
on the same talus_demo.py scenario (same source/target subjects, same random seeds).
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

src_stl = os.path.join(TALUS_DIR, "200001-xx-f-055xxxxxx-tal-l-c-d-s%.stl")
tgt_stl = os.path.join(TALUS_DIR, "200002-xx-m-068xxxxxx-tal-l-c-d-s%.stl")
config_path = os.path.join(REPO_ROOT, "configs", "liver.yaml")

checkpoints = {
    "pretrained (original)": os.path.join(REPO_ROOT, "snapshot", "liver_3D_1_one_transformer",
                                          "checkpoints", "model_best_loss.pth"),
    "talus fine-tuned":      os.path.join(REPO_ROOT, "snapshot", "talus_finetune",
                                          "checkpoints", "model_best_loss.pth"),
}

# build the scenario once so both checkpoints are evaluated on the identical pair/transform
np.random.seed(0)
src_pcd = stl_to_pcd(src_stl, n_points=8000, seed=0)
tgt_pcd_raw = stl_to_pcd(tgt_stl, n_points=8000, seed=1)

euler_gt = np.random.uniform(-np.pi / 3, np.pi / 3, size=3)
rot_gt = Rotation.from_euler('zyx', euler_gt).as_matrix().astype(np.float32)
trans_gt = (np.random.rand(3, 1) * 0.6 - 0.3).astype(np.float32)
tgt_pcd = (np.matmul(rot_gt, tgt_pcd_raw.T) + trans_gt).T

before_dist = chamfer_like(src_pcd, tgt_pcd)

config = load_config(config_path)
config = edict(config)
config.architecture = architectures[config.model_name]
config.device = torch.device('cuda:0')

demo_set = PairDemo(config, src_pcd, tgt_pcd)
neighborhood_limits = [19, 23, 29, 34]
loader, neighborhood_limits = get_dataloader(dataset=demo_set, batch_size=1, shuffle=False,
                                              num_workers=0, neighborhood_limits=neighborhood_limits)
list_data = demo_set.__getitem__(0)
inputs = collate_fn_descriptor([list_data], config, neighborhood_limits)

print(f"Source mesh: {os.path.basename(src_stl)}")
print(f"Target mesh: {os.path.basename(tgt_stl)}")
print(f"Mean NN dist src->tgt before alignment: {before_dist:.4f}\n")

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
    scores_vis_mask = scores_vis > th_score
    se_index = []
    for i in np.arange(len(scores_vis_mask)):
        if scores_vis_mask[i] and i in match_pred[:, 0]:
            idx = np.where(match_pred == i)[0][0]
            se_index.append(idx)
    match_pred_scores = match_pred[se_index, :]

    n_raw = len(match_pred)
    n_high_conf = len(match_pred_scores)
    if n_high_conf < 4:
        match_pred_scores = match_pred

    tsfm_pred = eva_regist(src_pcd, tgt_pcd, match_pred_scores, distance_threshold=0.15, ransac_n=4)
    rot_err_deg, trans_err = rot_trans_error(tsfm_pred, rot_gt, trans_gt)

    src_aligned = (np.matmul(tsfm_pred[:3, :3], src_pcd.T) + tsfm_pred[:3, 3:]).T
    after_dist = chamfer_like(src_aligned, tgt_pcd)

    print(f"=== {name} ===")
    print(f"  raw correspondences: {n_raw}, high-confidence (>{th_score}): {n_high_conf}")
    print(f"  rot_error_deg: {rot_err_deg:.3f}   trans_error: {trans_err:.4f}")
    print(f"  mean NN dist after alignment: {after_dist:.4f}\n")

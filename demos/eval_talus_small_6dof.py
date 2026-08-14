"""Same-bone partial-view registration test (like build_same_bone_scenario in
compare_talus_checkpoints.py) run over the first N bones in the talus_small dataset,
using the original pretrained checkpoint. Reports a per-axis 6DoF error (roll/pitch/yaw
rotation error in degrees + x/y/z translation error) for each bone, plus a before/after
plotly visualization per bone.
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from easydict import EasyDict as edict
from scipy.spatial.transform import Rotation
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib.util import load_config
from configs.models import architectures
from models.framework import KPFCNN
from datasets.dataloader import collate_fn_descriptor

from talus_demo import stl_to_pcd, eva_regist, chamfer_like, PairDemo

import warnings
warnings.filterwarnings("ignore")


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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TALUS_DIR = r"C:\Users\esun3\Documents\talus_small"
N_SUBJECTS = 5

config_path = os.path.join(REPO_ROOT, "configs", "liver.yaml")
config = load_config(config_path)
config = edict(config)
config.architecture = architectures[config.model_name]
config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {config.device}")

checkpoint_path = os.path.join(REPO_ROOT, "snapshot", "liver_3D_1_one_transformer",
                                "checkpoints", "model_best_loss.pth")

neighborhood_limits = [19, 23, 29, 34]
th_score = 0.9
out_csv = os.path.join(REPO_ROOT, "demos", "talus_small_6dof_results.csv")
viz_dir = os.path.join(REPO_ROOT, "demos", "talus_small_6dof_viz")
os.makedirs(viz_dir, exist_ok=True)


def build_same_bone_scenario(src_stl, seed):
    np.random.seed(seed)
    full_pcd = stl_to_pcd(src_stl, n_points=8000, seed=0)
    src_pcd = full_pcd
    tgt_partial = crop(full_pcd, p_keep=0.6)
    tgt_partial = tgt_partial + (np.random.rand(*tgt_partial.shape) - 0.5) * 0.02
    euler_gt = np.random.uniform(-np.pi / 3, np.pi / 3, size=3)
    rot_gt = Rotation.from_euler('zyx', euler_gt).as_matrix().astype(np.float32)
    trans_gt = (np.random.rand(3, 1) * 0.6 - 0.3).astype(np.float32)
    tgt_pcd = (np.matmul(rot_gt, tgt_partial.T) + trans_gt).T
    return src_pcd, tgt_pcd, rot_gt, trans_gt


def rot_trans_error_6dof(tsfm_pred, rot_gt, trans_gt):
    """Per-axis rotation error (roll/pitch/yaw, deg) and translation error (x/y/z)."""
    rot_pred = tsfm_pred[:3, :3]
    trans_pred = tsfm_pred[:3, 3]
    rot_err_mat = rot_pred @ rot_gt.T
    roll, pitch, yaw = Rotation.from_matrix(rot_err_mat).as_euler('xyz', degrees=True)
    tx, ty, tz = trans_pred - trans_gt.flatten()
    return abs(roll), abs(pitch), abs(yaw), abs(tx), abs(ty), abs(tz)


def save_visualization(fname, src_pcd, tgt_pcd, src_aligned, before_dist, after_dist, roll, pitch, yaw, tx, ty, tz):
    def point_trace(xyz, color, name, show_legend=True):
        return go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode='markers',
            marker=dict(size=2, color=color, opacity=0.8),
            name=name,
            legendgroup=name,
            showlegend=show_legend,
        )

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=(
            f"Before registration<br>mean NN dist={before_dist:.3f}",
            f"After registration<br>mean NN dist={after_dist:.3f}<br>"
            f"roll={roll:.1f} pitch={pitch:.1f} yaw={yaw:.1f} deg | "
            f"tx={tx:.3f} ty={ty:.3f} tz={tz:.3f}",
        ),
    )
    fig.add_trace(point_trace(src_pcd, 'red', 'source (full)'), row=1, col=1)
    fig.add_trace(point_trace(tgt_pcd, 'blue', 'target (partial, transformed)'), row=1, col=1)
    fig.add_trace(point_trace(src_aligned, 'green', 'source aligned to target'), row=1, col=2)
    fig.add_trace(point_trace(tgt_pcd, 'blue', 'target (partial, transformed)', show_legend=False), row=1, col=2)

    scene_kwargs = dict(aspectmode='data', camera=dict(eye=dict(x=1.4, y=1.4, z=1.2)))
    fig.update_layout(
        title=f"{fname}: same-bone partial-view registration",
        scene=scene_kwargs,
        scene2=scene_kwargs,
        legend=dict(orientation='h', y=-0.05),
        width=1300, height=700,
        margin=dict(l=10, r=10, t=90, b=10),
    )
    out_html = os.path.join(viz_dir, f"{os.path.splitext(fname)[0]}.html")
    fig.write_html(out_html, include_plotlyjs='cdn')
    return out_html


stl_files = sorted(f for f in os.listdir(TALUS_DIR) if f.lower().endswith(".stl"))[:N_SUBJECTS]

model = KPFCNN(config).to(config.device).eval()
state = torch.load(checkpoint_path, map_location=config.device)
model.load_state_dict(state['state_dict'])

rows = []
for i, fname in enumerate(stl_files):
    src_stl = os.path.join(TALUS_DIR, fname)
    t0 = time.time()
    print(f"[{i+1}/{len(stl_files)}] {fname}: sampling mesh...", flush=True)
    try:
        src_pcd, tgt_pcd, rot_gt, trans_gt = build_same_bone_scenario(src_stl, seed=i)
        print(f"    mesh sampled in {time.time()-t0:.1f}s, building inputs...", flush=True)

        t1 = time.time()
        demo_set = PairDemo(config, src_pcd, tgt_pcd)
        list_data = demo_set.__getitem__(0)
        inputs = collate_fn_descriptor([list_data], config, neighborhood_limits)
        print(f"    inputs collated in {time.time()-t1:.1f}s, running model...", flush=True)

        t2 = time.time()
        with torch.no_grad():
            dev_inputs = {}
            for k, v in inputs.items():
                dev_inputs[k] = [item.to(config.device) for item in v] if isinstance(v, list) else v.to(config.device)
            data = model(dev_inputs)
        print(f"    model ran in {time.time()-t2:.1f}s, running RANSAC...", flush=True)

        match_pred = data['match_pred'].detach().cpu()[:, 1:]
        scores_vis = data['scores_vis'].detach().cpu()

        vis_ok_ids = torch.nonzero(scores_vis > th_score, as_tuple=True)[0]
        keep = torch.isin(match_pred[:, 0], vis_ok_ids)
        match_pred_scores = match_pred[keep]
        if len(match_pred_scores) < 4:
            match_pred_scores = match_pred

        t3 = time.time()
        tsfm_pred = eva_regist(src_pcd, tgt_pcd, match_pred_scores, distance_threshold=0.15, ransac_n=4)
        roll, pitch, yaw, tx, ty, tz = rot_trans_error_6dof(tsfm_pred, rot_gt, trans_gt)
        print(f"    RANSAC done in {time.time()-t3:.1f}s, computing chamfer + viz...", flush=True)

        t4 = time.time()
        before_dist = chamfer_like(src_pcd, tgt_pcd)
        src_aligned = (np.matmul(tsfm_pred[:3, :3], src_pcd.T) + tsfm_pred[:3, 3:]).T
        after_dist = chamfer_like(src_aligned, tgt_pcd)
        out_html = save_visualization(fname, src_pcd, tgt_pcd, src_aligned, before_dist, after_dist,
                                       roll, pitch, yaw, tx, ty, tz)
        print(f"    chamfer + viz done in {time.time()-t4:.1f}s", flush=True)

        rows.append((fname, roll, pitch, yaw, tx, ty, tz))
        print(f"{fname:45s} roll={roll:7.3f} pitch={pitch:7.3f} yaw={yaw:7.3f}  "
              f"tx={tx:7.4f} ty={ty:7.4f} tz={tz:7.4f}  total={time.time()-t0:.1f}s  viz={out_html}", flush=True)
    except Exception as e:
        print(f"{fname:45s} FAILED: {e}", flush=True)

if rows:
    arr = np.array([r[1:] for r in rows])
    labels = ["roll_deg", "pitch_deg", "yaw_deg", "tx", "ty", "tz"]
    print(f"\n{len(rows)}/{len(stl_files)} bones succeeded")
    print("mean:  " + "  ".join(f"{l}={v:.4f}" for l, v in zip(labels, arr.mean(0))))
    print("std:   " + "  ".join(f"{l}={v:.4f}" for l, v in zip(labels, arr.std(0))))
    print("median:" + "  ".join(f"{l}={v:.4f}" for l, v in zip(labels, np.median(arr, 0))))

    with open(out_csv, "w") as f:
        f.write("bone,roll_deg,pitch_deg,yaw_deg,tx,ty,tz\n")
        for fname, roll, pitch, yaw, tx, ty, tz in rows:
            f.write(f"{fname},{roll},{pitch},{yaw},{tx},{ty},{tz}\n")
    print(f"\nSaved per-bone results to {out_csv}")

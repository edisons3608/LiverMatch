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
from talus_frames import frame_for_stl
from lib.talus_acs import error_6dof, axis_labels

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
    full_pcd, mm_per_unit = stl_to_pcd(src_stl, n_points=8000, seed=0, return_scale=True)
    src_pcd = full_pcd
    # frame from the FULL bone: registering a 60% crop to the template can land
    # in a wholly wrong pose (measured drifts of 99 and 178 deg)
    frame = frame_for_stl(src_stl, full_pcd, n_points=8000, mm_per_unit=mm_per_unit)
    tgt_partial = crop(full_pcd, p_keep=0.6)
    tgt_partial = tgt_partial + (np.random.rand(*tgt_partial.shape) - 0.5) * 0.02
    euler_gt = np.random.uniform(-np.pi / 3, np.pi / 3, size=3)
    rot_gt = Rotation.from_euler('zyx', euler_gt).as_matrix().astype(np.float32)
    trans_gt = (np.random.rand(3, 1) * 0.6 - 0.3).astype(np.float32)
    tgt_pcd = (np.matmul(rot_gt, tgt_partial.T) + trans_gt).T
    return src_pcd, tgt_pcd, rot_gt, trans_gt, frame


def rot_trans_error_6dof(tsfm_pred, rot_gt, trans_gt, frame=None):
    """Per-axis 6DoF error in the bone's anatomical frame.

    Every bone gets the same anatomical axes (propagated from one template),
    so these numbers are comparable across subjects -- world-frame ones are
    not, the raw scanner frames here differ by ~27 deg on average. Signed, and
    in mm when the frame carries a mm_per_unit.
    """
    return error_6dof(tsfm_pred, rot_gt, trans_gt, frame=frame, signed=True)


def _err_caption(roll, pitch, yaw, tx, ty, tz, frame):
    """6DoF error line for a plot title, labelled with the frame's own axes."""
    labels = axis_labels(frame)
    unit = 'mm' if frame is not None and frame.mm_per_unit is not None else 'units'
    rot = ' '.join(f'{l}={v:+.1f}' for l, v in zip(labels[:3], (roll, pitch, yaw)))
    trans = ' '.join(f'{l}={v:+.3f}' for l, v in zip(labels[3:], (tx, ty, tz)))
    return f'{rot} deg | {trans} {unit}'


def save_visualization(fname, src_pcd, tgt_pcd, src_aligned, before_dist, after_dist,
                        roll, pitch, yaw, tx, ty, tz, frame=None):
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
            + _err_caption(roll, pitch, yaw, tx, ty, tz, frame),
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
LAST_FRAME = None
for i, fname in enumerate(stl_files):
    src_stl = os.path.join(TALUS_DIR, fname)
    t0 = time.time()
    print(f"[{i+1}/{len(stl_files)}] {fname}: sampling mesh...", flush=True)
    try:
        src_pcd, tgt_pcd, rot_gt, trans_gt, frame = build_same_bone_scenario(src_stl, seed=i)
        LAST_FRAME = frame
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
        roll, pitch, yaw, tx, ty, tz = rot_trans_error_6dof(tsfm_pred, rot_gt, trans_gt, frame)
        print(f"    RANSAC done in {time.time()-t3:.1f}s, computing chamfer + viz...", flush=True)

        t4 = time.time()
        before_dist = chamfer_like(src_pcd, tgt_pcd)
        src_aligned = (np.matmul(tsfm_pred[:3, :3], src_pcd.T) + tsfm_pred[:3, 3:]).T
        after_dist = chamfer_like(src_aligned, tgt_pcd)
        out_html = save_visualization(fname, src_pcd, tgt_pcd, src_aligned, before_dist, after_dist,
                                       roll, pitch, yaw, tx, ty, tz, frame)
        print(f"    chamfer + viz done in {time.time()-t4:.1f}s", flush=True)

        rows.append((fname, roll, pitch, yaw, tx, ty, tz))
        print(f"{fname:45s} " + _err_caption(roll, pitch, yaw, tx, ty, tz, frame)
              + f"  total={time.time()-t0:.1f}s  viz={out_html}", flush=True)
    except Exception as e:
        print(f"{fname:45s} FAILED: {e}", flush=True)

TOTAL_N, ITEM_NAME = len(stl_files), 'bone'
if rows:
    arr = np.array([r[1:] for r in rows])
    labels = axis_labels(LAST_FRAME)
    unit = 'mm' if LAST_FRAME is not None and LAST_FRAME.mm_per_unit is not None else 'units'
    units = ['deg'] * 3 + [unit] * 3
    print(f"\n{len(rows)}/{TOTAL_N} {ITEM_NAME}s succeeded"
          f"  (errors in the {LAST_FRAME.source if LAST_FRAME else 'world'} frame)")
    # signed mean +/- std keeps a systematic bias distinguishable from scatter;
    # MAE is the magnitude summary that abs()-then-mean used to give
    for j, (l, u) in enumerate(zip(labels, units)):
        v = arr[:, j]
        print(f"  {l + ' [' + u + ']':<20s} mean={v.mean():+9.4f}  std={v.std():8.4f}  "
              f"MAE={np.abs(v).mean():8.4f}  median={np.median(v):+9.4f}")

    with open(out_csv, "w") as f:
        f.write("bone," + ",".join(l + "_" + u for l, u in zip(labels, units)) + "\n")
        for fname, roll, pitch, yaw, tx, ty, tz in rows:
            f.write(f"{fname},{roll},{pitch},{yaw},{tx},{ty},{tz}\n")
    print(f"\nSaved per-bone results to {out_csv}")

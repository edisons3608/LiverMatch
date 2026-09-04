"""Run the existing pretrained checkpoint (no paint-specific training) on synthetic
paint_talus samples and report per-sample 6DoF registration error, plus a before/after
visualization for each sample.
"""
import os
import sys
import time
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import open3d as o3d
from easydict import EasyDict as edict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib.util import load_config
from configs.models import architectures
from models.framework import KPFCNN
from datasets.dataloader import collate_fn_descriptor
from datasets.paint_talus import paintTalusDataset

from talus_demo import eva_regist, chamfer_like
from talus_frames import frame_for, mm_per_unit_from_cache
from lib.talus_acs import error_6dof, axis_labels

import warnings
warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_SAMPLES = 5
NEIGHBORHOOD_LIMITS = [19, 23, 29, 34]
TH_SCORE = 0.9

parser = argparse.ArgumentParser()
parser.add_argument('--config', default=os.path.join(REPO_ROOT, 'configs', 'talus_paint_10ep.yaml'))
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--out-csv', default=os.path.join(REPO_ROOT, 'demos', 'paint_talus_pretrained_6dof_results.csv'))
parser.add_argument('--viz-dir', default=os.path.join(REPO_ROOT, 'demos', 'paint_talus_pretrained_viz'))
args = parser.parse_args()

config_path = args.config
config = edict(load_config(config_path))
config.architecture = architectures[config.model_name]
config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {config.device}")

checkpoint_path = args.checkpoint if args.checkpoint is not None else os.path.join(REPO_ROOT, config.pretrain)
out_csv = args.out_csv
viz_dir = args.viz_dir
os.makedirs(viz_dir, exist_ok=True)


def rot_trans_error_6dof(tsfm_pred, rot_gt, trans_gt, frame=None):
    """Per-axis 6DoF error in the bone's anatomical frame.

    Every bone gets the same anatomical axes (propagated from one template), so
    these are comparable across samples -- world-frame ones are not, since the
    raw scanner frames here differ by ~27 deg on average. Signed, and in mm when
    the frame carries a mm_per_unit.
    """
    return error_6dof(tsfm_pred, rot_gt, trans_gt, frame=frame, signed=True)


def _err_caption(roll, pitch, yaw, tx, ty, tz, frame):
    """6DoF error line for a plot title, labelled with the frame's own axes."""
    labels = axis_labels(frame)
    unit = 'mm' if frame is not None and frame.mm_per_unit is not None else 'units'
    rot = ' '.join(f'{l}={v:+.1f}' for l, v in zip(labels[:3], (roll, pitch, yaw)))
    trans = ' '.join(f'{l}={v:+.3f}' for l, v in zip(labels[3:], (tx, ty, tz)))
    return f'{rot} deg | {trans} {unit}'


def point_trace(xyz, color, name, size=2, opacity=0.8, show_legend=True):
    return go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode='markers',
        marker=dict(size=size, color=color, opacity=opacity),
        name=name,
        legendgroup=name,
        showlegend=show_legend,
    )


def line_trace(xyz, color, name, width=5, opacity=0.95, show_legend=True):
    return go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode='lines',
        connectgaps=False,
        line=dict(width=width, color=color),
        opacity=opacity,
        name=name,
        legendgroup=name,
        showlegend=show_legend,
    )


def save_visualization(fname, src_pcd, tgt_pcd, src_aligned, strokes_src, strokes_tgt,
                        before_dist, after_dist, roll, pitch, yaw, tx, ty, tz, frame=None):
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=(
            f"Before registration<br>mean NN dist={before_dist:.4f}",
            f"After registration<br>mean NN dist={after_dist:.4f}<br>"
            + _err_caption(roll, pitch, yaw, tx, ty, tz, frame),
        ),
    )
    fig.add_trace(point_trace(src_pcd, 'lightgray', 'source (full)', opacity=0.35), row=1, col=1)
    fig.add_trace(point_trace(tgt_pcd, 'royalblue', 'target (painted, transformed)'), row=1, col=1)
    for s in strokes_tgt:
        fig.add_trace(line_trace(s, 'darkorange', 'target stroke', show_legend=False), row=1, col=1)

    fig.add_trace(point_trace(src_aligned, 'green', 'source aligned to target'), row=1, col=2)
    fig.add_trace(point_trace(tgt_pcd, 'royalblue', 'target (painted, transformed)', show_legend=False), row=1, col=2)
    for s in strokes_tgt:
        fig.add_trace(line_trace(s, 'darkorange', 'target stroke', show_legend=False), row=1, col=2)

    scene_kwargs = dict(aspectmode='data', camera=dict(eye=dict(x=1.4, y=1.4, z=1.2)))
    fig.update_layout(
        title=f"{fname}: pretrained-model registration on paint_talus sample",
        scene=scene_kwargs,
        scene2=scene_kwargs,
        legend=dict(orientation='h', y=-0.05),
        width=1300, height=700,
        margin=dict(l=10, r=10, t=90, b=10),
    )
    out_html = os.path.join(viz_dir, f"{fname}.html")
    fig.write_html(out_html, include_plotlyjs='cdn')
    return out_html


dataset = paintTalusDataset(config, "val")
n_samples = min(N_SAMPLES, len(dataset.file_list))

model = KPFCNN(config).to(config.device).eval()
state = torch.load(checkpoint_path, map_location=config.device)
model.load_state_dict(state['state_dict'])
print(f"Loaded pretrained checkpoint: {checkpoint_path}")

rows = []
LAST_FRAME = None
for i in range(n_samples):
    stem = os.path.splitext(os.path.basename(dataset.file_list[i]))[0]
    t0 = time.time()
    print(f"[{i+1}/{n_samples}] {stem}: generating sample...", flush=True)
    try:
        sample, meta = dataset.get_input_train(i, return_meta=True)
        src_pcd, tgt_pcd, src_feats, tgt_feats, rot_gt, trans_gt, correspondences, src_pcd_raw, tgt_pcd_raw, _ = sample

        # src_pcd is the full bone (the target is the painted partial view), which is
        # what the frame has to be derived from -- a partial view registers unreliably
        sample_path = dataset._resolve_sample_path(i)
        frame = frame_for(stem, src_pcd, mm_per_unit=mm_per_unit_from_cache(sample_path))
        LAST_FRAME = frame

        t1 = time.time()
        inputs = collate_fn_descriptor([sample], config, NEIGHBORHOOD_LIMITS)
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

        vis_ok_ids = torch.nonzero(scores_vis > TH_SCORE, as_tuple=True)[0]
        keep = torch.isin(match_pred[:, 0], vis_ok_ids)
        match_pred_scores = match_pred[keep]
        if len(match_pred_scores) < 4:
            match_pred_scores = match_pred

        t3 = time.time()
        # Bound RANSAC iterations - Open3D's default (max_iteration=100000) can take minutes
        # on noisy synthetic correspondence sets.
        ransac_criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(4000, 0.999)
        tsfm_pred = eva_regist(src_pcd, tgt_pcd, match_pred_scores.numpy(), distance_threshold=0.15, ransac_n=4, criteria=ransac_criteria)
        roll, pitch, yaw, tx, ty, tz = rot_trans_error_6dof(tsfm_pred, rot_gt, trans_gt, frame)
        print(f"    RANSAC done in {time.time()-t3:.1f}s, computing chamfer + viz...", flush=True)

        t4 = time.time()
        before_dist = chamfer_like(src_pcd, tgt_pcd)
        src_aligned = (np.matmul(tsfm_pred[:3, :3], src_pcd.T) + tsfm_pred[:3, 3:]).T
        after_dist = chamfer_like(src_aligned, tgt_pcd)
        out_html = save_visualization(stem, src_pcd, tgt_pcd, src_aligned,
                                       meta["strokes_src"], meta["strokes_tgt"],
                                       before_dist, after_dist, roll, pitch, yaw, tx, ty, tz, frame)
        print(f"    chamfer + viz done in {time.time()-t4:.1f}s", flush=True)

        rows.append((stem, roll, pitch, yaw, tx, ty, tz))
        print(f"{stem:45s} " + _err_caption(roll, pitch, yaw, tx, ty, tz, frame)
              + f"  total={time.time()-t0:.1f}s  viz={out_html}", flush=True)
    except Exception as e:
        print(f"{stem:45s} FAILED: {e}", flush=True)

if rows:
    arr = np.array([r[1:] for r in rows])
    labels = axis_labels(LAST_FRAME)
    unit = 'mm' if LAST_FRAME is not None and LAST_FRAME.mm_per_unit is not None else 'units'
    units = ['deg'] * 3 + [unit] * 3
    print(f"\n{len(rows)}/{n_samples} samples succeeded"
          f"  (errors in the {LAST_FRAME.source if LAST_FRAME else 'world'} frame)")
    # signed mean +/- std keeps a systematic bias distinguishable from scatter;
    # MAE is the magnitude summary that abs()-then-mean used to give
    for j, (l, u) in enumerate(zip(labels, units)):
        v = arr[:, j]
        print(f"  {l + ' [' + u + ']':<20s} mean={v.mean():+9.4f}  std={v.std():8.4f}  "
              f"MAE={np.abs(v).mean():8.4f}  median={np.median(v):+9.4f}")

    with open(out_csv, "w") as f:
        f.write("sample," + ",".join(l + "_" + u for l, u in zip(labels, units)) + "\n")
        for stem, roll, pitch, yaw, tx, ty, tz in rows:
            f.write(f"{stem},{roll},{pitch},{yaw},{tx},{ty},{tz}\n")
    print(f"\nSaved per-sample results to {out_csv}")

"""Before/after registration visualizations for the talus_finetune_selfpair_30ep checkpoint,
over the same paired trials used in compare_talus_checkpoints.py (same seeds), so the plots
line up with the numbers already reported for that checkpoint.
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from easydict import EasyDict as edict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib.util import load_config
from configs.models import architectures
from models.framework import KPFCNN
from datasets.dataloader import collate_fn_descriptor

from talus_demo import eva_regist, rot_trans_error, chamfer_like, PairDemo
from compare_talus_checkpoints import (
    build_cross_subject_trials, build_same_bone_trials, RANSAC_CRITERIA, th_score,
    neighborhood_limits as NEIGHBORHOOD_LIMITS,
)

import warnings
warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument('--n-trials', type=int, default=5)
parser.add_argument('--seed-base', type=int, default=100)
parser.add_argument('--scenario', choices=['same-bone', 'cross-subject', 'both'], default='same-bone')
parser.add_argument('--checkpoint', default=os.path.join(REPO_ROOT, 'snapshot', 'talus_finetune_selfpair_30ep',
                                                          'checkpoints', 'model_best_loss.pth'))
parser.add_argument('--out-dir', default=os.path.join(REPO_ROOT, 'demos', 'talus_finetune_30ep_viz'))
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

config_path = os.path.join(REPO_ROOT, "configs", "liver.yaml")
config = load_config(config_path)
config = edict(config)
config.architecture = architectures[config.model_name]
config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {config.device}")

model = KPFCNN(config).to(config.device).eval()
state = torch.load(args.checkpoint, map_location=config.device)
model.load_state_dict(state['state_dict'])
print(f"Loaded checkpoint: {args.checkpoint}")


def point_trace(xyz, color, name, size=2, opacity=0.8, show_legend=True):
    return go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode='markers',
        marker=dict(size=size, color=color, opacity=opacity),
        name=name,
        legendgroup=name,
        showlegend=show_legend,
    )


def save_visualization(fname, src_pcd, tgt_pcd, src_aligned, before_dist, after_dist, rot_err, trans_err):
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=(
            f"Before registration<br>mean NN dist={before_dist:.4f}",
            f"After registration<br>mean NN dist={after_dist:.4f}<br>"
            f"rot_err={rot_err:.2f} deg | trans_err={trans_err:.4f}",
        ),
    )
    fig.add_trace(point_trace(src_pcd, 'lightgray', 'source (full)', opacity=0.35), row=1, col=1)
    fig.add_trace(point_trace(tgt_pcd, 'royalblue', 'target (transformed)'), row=1, col=1)

    fig.add_trace(point_trace(src_aligned, 'green', 'source aligned to target'), row=1, col=2)
    fig.add_trace(point_trace(tgt_pcd, 'royalblue', 'target (transformed)', show_legend=False), row=1, col=2)

    scene_kwargs = dict(aspectmode='data', camera=dict(eye=dict(x=1.4, y=1.4, z=1.2)))
    fig.update_layout(
        title=f"{fname}: talus_finetune_selfpair_30ep registration",
        scene=scene_kwargs,
        scene2=scene_kwargs,
        legend=dict(orientation='h', y=-0.05),
        width=1300, height=700,
        margin=dict(l=10, r=10, t=90, b=10),
    )
    out_html = os.path.join(args.out_dir, f"{fname}.html")
    fig.write_html(out_html, include_plotlyjs='cdn')
    return out_html


def run_and_visualize(scenario_key, desc, trials):
    print(f"\n########## {scenario_key} ##########\n{desc}")
    rows = []
    for t_i, (src_pcd, tgt_pcd, rot_gt, trans_gt) in enumerate(trials):
        before_dist = chamfer_like(src_pcd, tgt_pcd)
        demo_set = PairDemo(config, src_pcd, tgt_pcd)
        list_data = demo_set.__getitem__(0)
        inputs = collate_fn_descriptor([list_data], config, NEIGHBORHOOD_LIMITS)

        with torch.no_grad():
            dev_inputs = {}
            for k, v in inputs.items():
                dev_inputs[k] = [item.to(config.device) for item in v] if isinstance(v, list) else v.to(config.device)
            data = model(dev_inputs)

        match_pred = data['match_pred'].detach().cpu()[:, 1:]
        scores_vis = data['scores_vis'].detach().cpu()
        vis_ok_ids = torch.nonzero(scores_vis > th_score, as_tuple=True)[0]
        keep = torch.isin(match_pred[:, 0], vis_ok_ids)
        match_pred_scores = match_pred[keep]
        if len(match_pred_scores) < 4:
            match_pred_scores = match_pred

        tsfm_pred = eva_regist(src_pcd, tgt_pcd, match_pred_scores.numpy(), distance_threshold=0.15,
                                ransac_n=4, criteria=RANSAC_CRITERIA)
        rot_err, trans_err = rot_trans_error(tsfm_pred, rot_gt, trans_gt)

        src_aligned = (np.matmul(tsfm_pred[:3, :3], src_pcd.T) + tsfm_pred[:3, 3:]).T
        after_dist = chamfer_like(src_aligned, tgt_pcd)

        fname = f"{scenario_key}_trial{t_i+1}"
        out_html = save_visualization(fname, src_pcd, tgt_pcd, src_aligned, before_dist, after_dist,
                                       rot_err, trans_err)
        rows.append(out_html)
        print(f"  trial {t_i+1}/{len(trials)}: rot_err={rot_err:6.2f} deg  trans_err={trans_err:.4f}  "
              f"before={before_dist:.4f} after={after_dist:.4f}  viz={out_html}")
    return rows


all_viz = []
if args.scenario in ('same-bone', 'both'):
    desc, trials = build_same_bone_trials(args.n_trials, args.seed_base + 1000)
    all_viz += run_and_visualize('same-bone', desc, trials)
if args.scenario in ('cross-subject', 'both'):
    desc, trials = build_cross_subject_trials(args.n_trials, args.seed_base)
    all_viz += run_and_visualize('cross-subject', desc, trials)

print(f"\nSaved {len(all_viz)} visualization(s) to {args.out_dir}")
for p in all_viz:
    print(f"  {p}")

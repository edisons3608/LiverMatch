"""
Try the pretrained LiverMatch (liver point-cloud) correspondence network on a completely
different anatomy: talus bone surfaces from an SSM dataset of STL meshes.

This is an out-of-domain experiment (bone vs. soft-tissue liver, inter-subject shape
matching vs. intra-patient deformation matching), so treat the numbers as a sanity check
of generalization, not a benchmark result. See README's "Important note to train and test
on other datasets" section: point clouds must be normalized to roughly the same scale the
network was trained at (KPConv's radii are absolute, not scale-invariant).

Pipeline:
  1. Load two different subjects' talus STL meshes, surface-sample them into point clouds,
     and rescale each to match the liver dataset's bounding-box-diagonal convention.
  2. Apply a *known* random rigid transform to the target cloud (this stands in for the
     unknown pose a laparoscope/other sensor would observe from), so we have ground truth
     to evaluate the recovered pose against.
  3. Run the pretrained matching network to get correspondences, then RANSAC to recover a
     rigid transform from source -> target frame.
  4. Compare the recovered transform to the known one (rotation/translation error), and
     save a before/after 3D plot.
"""
import os.path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset
from easydict import EasyDict as edict
from scipy.spatial.transform import Rotation

from lib.util import load_config
from configs.models import architectures
from models.framework import KPFCNN
from datasets.dataloader import get_dataloader, collate_fn_descriptor

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser

import warnings
warnings.filterwarnings("ignore")


def stl_to_pcd(path, n_points=8000, target_diag=3.49, seed=0):
    """Surface-sample an STL mesh into a point cloud, centered and rescaled so its
    bounding-box diagonal matches the liver dataset's typical scaled extent (~3.49,
    see test_data/Liver1/*.ply after the demo's /100 normalization)."""
    o3d.utility.random.seed(seed)
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(number_of_points=n_points)
    pts = np.asarray(pcd.points)

    diag = np.linalg.norm(pts.max(0) - pts.min(0))
    pts = pts * (target_diag / diag)
    pts = pts - pts.mean(0)
    return pts.astype(np.float32)


def to_o3d_pcd(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz))
    return pcd


def eva_regist(src_pcd, tgt_pcd, corrs, distance_threshold=0.05, ransac_n=4):
    src_o3d = to_o3d_pcd(src_pcd)
    tgt_o3d = to_o3d_pcd(tgt_pcd)
    corrs = np.asarray(corrs).astype(np.int32)
    corrs_o3d = o3d.utility.Vector2iVector(corrs)

    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source=src_o3d, target=tgt_o3d, corres=corrs_o3d,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
    )
    return np.array(result_ransac.transformation)


def rot_trans_error(tsfm_pred, rot_gt, trans_gt):
    rot_pred = tsfm_pred[:3, :3]
    trans_pred = tsfm_pred[:3, 3]

    cos_theta = (np.trace(rot_pred.T @ rot_gt) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    rot_err_deg = np.degrees(np.arccos(cos_theta))
    trans_err = np.linalg.norm(trans_pred - trans_gt.flatten())
    return rot_err_deg, trans_err


def chamfer_like(a, b):
    """mean one-directional nearest-neighbor distance a -> b (cheap proxy, not full Chamfer)."""
    tree = o3d.geometry.KDTreeFlann(to_o3d_pcd(b))
    dists = []
    for p in a:
        _, _, d2 = tree.search_knn_vector_3d(p, 1)
        dists.append(np.sqrt(d2[0]))
    return float(np.mean(dists))


class PairDemo(Dataset):
    def __init__(self, config, src_pcd, tgt_pcd):
        self.config = config
        self.src_pcd = src_pcd
        self.tgt_pcd = tgt_pcd

    def __len__(self):
        return 1

    def __getitem__(self, item):
        src_feats = np.ones_like(self.src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(self.tgt_pcd[:, :1]).astype(np.float32)
        rot = np.eye(3).astype(np.float32)
        trans = np.ones((3, 1)).astype(np.float32)
        correspondences = torch.ones(1, 2).long()
        return (self.src_pcd, self.tgt_pcd, src_feats, tgt_feats, rot, trans,
                correspondences, self.src_pcd, self.tgt_pcd, torch.ones(1))


if __name__ == '__main__':
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TALUS_DIR = r"C:\Users\esun3\Documents\talus_small"

    src_stl = os.path.join(TALUS_DIR, "200001-xx-f-055xxxxxx-tal-l-c-d-s%.stl")
    tgt_stl = os.path.join(TALUS_DIR, "200002-xx-m-068xxxxxx-tal-l-c-d-s%.stl")

    config_path = os.path.join(REPO_ROOT, "configs", "liver.yaml")
    pretrain_path = os.path.join(REPO_ROOT, "snapshot", "liver_3D_1_one_transformer",
                                  "checkpoints", "model_best_loss.pth")
    out_html = os.path.join(REPO_ROOT, "demos", "talus_demo_result.html")

    print(f"Source mesh: {os.path.basename(src_stl)}")
    print(f"Target mesh: {os.path.basename(tgt_stl)}")

    np.random.seed(0)
    src_pcd = stl_to_pcd(src_stl, n_points=8000, seed=0)
    tgt_pcd_raw = stl_to_pcd(tgt_stl, n_points=8000, seed=1)

    # Apply a *known* random rigid transform to the target, standing in for the
    # unknown pose a real sensor would present. We'll evaluate against this later.
    euler_gt = np.random.uniform(-np.pi / 3, np.pi / 3, size=3)  # up to 60 deg/axis
    rot_gt = Rotation.from_euler('zyx', euler_gt).as_matrix().astype(np.float32)
    trans_gt = (np.random.rand(3, 1) * 0.6 - 0.3).astype(np.float32)  # +/- 0.3 units
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

    model = KPFCNN(config).to(config.device).eval()
    state = torch.load(pretrain_path, map_location=config.device)
    model.load_state_dict(state['state_dict'])

    list_data = demo_set.__getitem__(0)
    inputs = collate_fn_descriptor([list_data], config, neighborhood_limits)
    with torch.no_grad():
        for k, v in inputs.items():
            inputs[k] = [item.to(config.device) for item in v] if isinstance(v, list) else v.to(config.device)
        data = model(inputs)

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

    print(f"\nRaw correspondences found: {len(match_pred)}")
    print(f"High-confidence (score>{th_score}) correspondences: {len(match_pred_scores)}")

    if len(match_pred_scores) < 4:
        print("Too few high-confidence matches for RANSAC; falling back to all raw matches.")
        match_pred_scores = match_pred

    tsfm_pred = eva_regist(src_pcd, tgt_pcd, match_pred_scores, distance_threshold=0.15, ransac_n=4)
    rot_err_deg, trans_err = rot_trans_error(tsfm_pred, rot_gt, trans_gt)

    src_aligned = (np.matmul(tsfm_pred[:3, :3], src_pcd.T) + tsfm_pred[:3, 3:]).T
    after_dist = chamfer_like(src_aligned, tgt_pcd)

    print("\n  " + ("{:>22} | " * 3).format("rot_error_deg", "trans_error", "n_matches_used"))
    print(("&{: 22.3f}  " * 2 + "&{:22d}").format(rot_err_deg, trans_err, len(match_pred_scores)))
    print(f"\nMean NN dist src->tgt  before alignment: {before_dist:.4f}")
    print(f"Mean NN dist src->tgt  after  alignment: {after_dist:.4f}")

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
            f"After registration<br>rot_err={rot_err_deg:.1f} deg, trans_err={trans_err:.3f}<br>mean NN dist={after_dist:.3f}",
        ),
    )
    fig.add_trace(point_trace(src_pcd, 'red', 'source (subj 1)'), row=1, col=1)
    fig.add_trace(point_trace(tgt_pcd, 'blue', 'target (subj 2, transformed)'), row=1, col=1)
    fig.add_trace(point_trace(src_aligned, 'green', 'source aligned to target'), row=1, col=2)
    fig.add_trace(point_trace(tgt_pcd, 'blue', 'target (subj 2, transformed)', show_legend=False), row=1, col=2)

    scene_kwargs = dict(aspectmode='data', camera=dict(eye=dict(x=1.4, y=1.4, z=1.2)))
    fig.update_layout(
        title="Talus demo: LiverMatch correspondences + RANSAC registration (drag to rotate, scroll to zoom)",
        scene=scene_kwargs,
        scene2=scene_kwargs,
        legend=dict(orientation='h', y=-0.05),
        width=1300, height=700,
        margin=dict(l=10, r=10, t=90, b=10),
    )

    fig.write_html(out_html, include_plotlyjs='cdn')
    print(f"\nSaved interactive visualization to {out_html}")
    try:
        webbrowser.open(f"file://{out_html}")
    except Exception:
        pass

    out_npz = os.path.join(REPO_ROOT, "demos", "talus_demo_points.npz")
    np.savez(out_npz,
             src_pcd=src_pcd, tgt_pcd=tgt_pcd, src_aligned=src_aligned,
             rot_err_deg=rot_err_deg, trans_err=trans_err,
             before_dist=before_dist, after_dist=after_dist,
             n_matches_raw=len(match_pred), n_matches_used=len(match_pred_scores))
    print(f"Saved point arrays to {out_npz}")

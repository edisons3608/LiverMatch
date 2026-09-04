"""Compare the original pretrained LiverMatch checkpoint against the talus fine-tune(s)
on two scenarios:
  1. Cross-subject (same as talus_demo.py): two different subjects' full bones.
  2. Same-bone partial view: one subject's bone, cropped/noised, under a known rigid
     transform -- the actual target use case (registering the same bone to itself).

Each scenario is run over N_TRIALS random transforms (and, for same-bone, random crop
directions). The same set of trial seeds is used for every checkpoint, so each checkpoint
sees exactly the same inputs -- this makes the comparison paired (removes cross-checkpoint
scenario noise) and, combined with averaging over trials, damps the run-to-run variance
introduced by Open3D's RANSAC (which is stochastic and unseedable via this API). Report
mean +/- std per checkpoint instead of a single noisy sample.
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import open3d as o3d
import torch
from easydict import EasyDict as edict
from scipy.spatial.transform import Rotation

from lib.util import load_config
from configs.models import architectures
from models.framework import KPFCNN
from datasets.dataloader import collate_fn_descriptor, get_dataloader

from talus_demo import stl_to_pcd, eva_regist, rot_trans_error, chamfer_like, PairDemo
from talus_frames import frame_for_stl
from lib.talus_acs import error_6dof, axis_labels

import warnings
warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TALUS_DIR = r"C:\Users\esun3\Documents\talus_small"

config_path = os.path.join(REPO_ROOT, "configs", "liver.yaml")
config = load_config(config_path)
config = edict(config)
config.architecture = architectures[config.model_name]
config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

checkpoints = {
    "pretrained (original)":         os.path.join(REPO_ROOT, "snapshot", "liver_3D_1_one_transformer",
                                                  "checkpoints", "model_best_loss.pth"),
    "talus fine-tuned (9ep self-pair)":  os.path.join(REPO_ROOT, "snapshot", "talus_finetune_selfpair",
                                                  "checkpoints", "model_best_loss.pth"),
    "talus fine-tuned (30ep self-pair)": os.path.join(REPO_ROOT, "snapshot", "talus_finetune_selfpair_30ep",
                                                  "checkpoints", "model_best_loss.pth"),
    "talus fine-tuned (30ep self-pair, ds3)": os.path.join(REPO_ROOT, "snapshot", "talus_finetune_selfpair_ds3",
                                                  "checkpoints", "model_best_loss.pth"),
}

neighborhood_limits = [19, 23, 29, 34]
th_score = 0.9
# Bound RANSAC iterations -- Open3D's default (max_iteration=100000) can take a very long
# time, and with N_TRIALS x N_scenarios x N_checkpoints RANSAC calls that adds up fast.
RANSAC_CRITERIA = o3d.pipelines.registration.RANSACConvergenceCriteria(4000, 0.999)


def uniform_2_sphere(rng):
    phi = rng.uniform(0.0, 2 * np.pi)
    cos_theta = rng.uniform(-1.0, 1.0)
    theta = np.arccos(cos_theta)
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def crop(points, p_keep, rng):
    rand_xyz = uniform_2_sphere(rng)
    centroid = points.mean(0)
    dist = np.dot(points - centroid, rand_xyz)
    mask = dist > np.percentile(dist, (1.0 - p_keep) * 100)
    return points[mask]


def random_rigid(rng):
    euler_gt = rng.uniform(-np.pi / 3, np.pi / 3, size=3)
    rot_gt = Rotation.from_euler('zyx', euler_gt).as_matrix().astype(np.float32)
    trans_gt = (rng.random((3, 1)) * 0.6 - 0.3).astype(np.float32)
    return rot_gt, trans_gt


def rot_trans_error_6dof(tsfm_pred, rot_gt, trans_gt, frame=None):
    """Per-axis 6DoF error, expressed in the bone's anatomical frame when given.

    Without a frame these are world-axis Euler angles and a world-origin
    translation difference, which are NOT comparable between subjects: the raw
    scanner frames in this dataset differ by ~27 deg on average (max 43), so each
    subject's tx points somewhere else anatomically. Pass the source bone's
    AnatomicalFrame and the axes mean the same thing for every subject.

    Values are signed -- a consistent bias and symmetric scatter are different
    failures, and abs() collapses them into the same number. Translations come
    back in mm when the frame carries a mm_per_unit.
    """
    return error_6dof(tsfm_pred, rot_gt, trans_gt, frame=frame, signed=True)


def build_cross_subject_trials(n_trials, seed_base, n_points=8000):
    src_stl = os.path.join(TALUS_DIR, "200001-xx-f-055xxxxxx-tal-l-c-d-s%.stl")
    tgt_stl = os.path.join(TALUS_DIR, "200002-xx-m-068xxxxxx-tal-l-c-d-s%.stl")
    src_pcd, mm_per_unit = stl_to_pcd(src_stl, n_points=n_points, seed=0, return_scale=True)
    tgt_pcd_raw = stl_to_pcd(tgt_stl, n_points=n_points, seed=1)
    frame = frame_for_stl(src_stl, src_pcd, n_points=n_points, mm_per_unit=mm_per_unit)
    desc = (f"Cross-subject: {os.path.basename(src_stl)} -> {os.path.basename(tgt_stl)} "
            f"({n_points} pts/side); errors in the {frame.source} frame "
            f"(fitness={frame.fitness:.3f}, rmse={frame.rmse:.4f})")

    trials = []
    for i in range(n_trials):
        rng = np.random.default_rng(seed_base + i)
        rot_gt, trans_gt = random_rigid(rng)
        tgt_pcd = (np.matmul(rot_gt, tgt_pcd_raw.T) + trans_gt).T
        trials.append((src_pcd, tgt_pcd, rot_gt, trans_gt, frame))
    return desc, trials


def build_same_bone_trials(n_trials, seed_base, n_points=8000):
    src_stl = os.path.join(TALUS_DIR, "200001-xx-f-055xxxxxx-tal-l-c-d-s%.stl")
    full_pcd, mm_per_unit = stl_to_pcd(src_stl, n_points=n_points, seed=0, return_scale=True)
    # the frame is derived from the FULL bone: registering a 60% crop to the template
    # can land in a wholly wrong pose (measured drifts of 99 and 178 deg)
    frame = frame_for_stl(src_stl, full_pcd, n_points=n_points, mm_per_unit=mm_per_unit)
    desc = (f"Same-bone partial view: {os.path.basename(src_stl)} (60% visible, +noise, "
            f"{n_points} pts source); errors in the {frame.source} frame "
            f"(fitness={frame.fitness:.3f}, rmse={frame.rmse:.4f})")

    trials = []
    for i in range(n_trials):
        rng = np.random.default_rng(seed_base + i)
        tgt_partial = crop(full_pcd, p_keep=0.6, rng=rng)
        tgt_partial = tgt_partial + (rng.random(tgt_partial.shape) - 0.5) * 0.02
        rot_gt, trans_gt = random_rigid(rng)
        tgt_pcd = (np.matmul(rot_gt, tgt_partial.T) + trans_gt).T
        trials.append((full_pcd, tgt_pcd, rot_gt, trans_gt, frame))
    return desc, trials


def run_trial(model, src_pcd, tgt_pcd, rot_gt, trans_gt, frame=None):
    before_dist = chamfer_like(src_pcd, tgt_pcd)
    demo_set = PairDemo(config, src_pcd, tgt_pcd)
    list_data = demo_set.__getitem__(0)
    inputs = collate_fn_descriptor([list_data], config, neighborhood_limits)

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

    n_raw = len(match_pred)
    n_high_conf = len(match_pred_scores)
    if n_high_conf < 4:
        match_pred_scores = match_pred

    tsfm_pred = eva_regist(src_pcd, tgt_pcd, match_pred_scores.numpy(), distance_threshold=0.15,
                            ransac_n=4, criteria=RANSAC_CRITERIA)
    rot_err_deg, trans_err = rot_trans_error(tsfm_pred, rot_gt, trans_gt)
    roll, pitch, yaw, tx, ty, tz = rot_trans_error_6dof(tsfm_pred, rot_gt, trans_gt, frame)

    src_aligned = (np.matmul(tsfm_pred[:3, :3], src_pcd.T) + tsfm_pred[:3, 3:]).T
    after_dist = chamfer_like(src_aligned, tgt_pcd)

    return dict(before_dist=before_dist, after_dist=after_dist, rot_err=rot_err_deg,
                trans_err=trans_err, n_raw=n_raw, n_high_conf=n_high_conf,
                roll=roll, pitch=pitch, yaw=yaw, tx=tx, ty=ty, tz=tz)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=5)
    parser.add_argument('--seed-base', type=int, default=100)
    parser.add_argument('--n-points', type=int, default=8000,
                         help='points sampled per side (source and target) for both scenarios; '
                              'the same-bone target is a crop of the source so it inherits this density too')
    parser.add_argument('--verbose', action='store_true', help='print every trial, not just the summary')
    args = parser.parse_args()

    scenarios = {
        "cross-subject (talus_demo.py)": build_cross_subject_trials(args.n_trials, args.seed_base,
                                                                      n_points=args.n_points),
        "same-bone partial view":        build_same_bone_trials(args.n_trials, args.seed_base + 1000,
                                                                  n_points=args.n_points),
    }

    for scenario_name, (desc, trials) in scenarios.items():
        print(f"\n########## {scenario_name} ##########")
        print(desc)
        print(f"{args.n_trials} trials, shared across checkpoints (paired comparison)")

        for name, ckpt_path in checkpoints.items():
            model = KPFCNN(config).to(config.device).eval()
            state = torch.load(ckpt_path, map_location=config.device)
            model.load_state_dict(state['state_dict'])

            rows = []
            labels = axis_labels(trials[0][4])
            rot_lbl, trans_lbl = labels[:3], labels[3:]
            unit = 'mm' if trials[0][4].mm_per_unit is not None else 'units'
            for t_i, (src_pcd, tgt_pcd, rot_gt, trans_gt, frame) in enumerate(trials):
                r = run_trial(model, src_pcd, tgt_pcd, rot_gt, trans_gt, frame)
                rows.append(r)
                if args.verbose:
                    print(f"    [{name}] trial {t_i+1}/{args.n_trials}: "
                          f"roll={r['roll']:6.2f} pitch={r['pitch']:6.2f} yaw={r['yaw']:6.2f}  "
                          f"tx={r['tx']:.4f} ty={r['ty']:.4f} tz={r['tz']:.4f}  "
                          f"corr={r['n_raw']}/{r['n_high_conf']} after_dist={r['after_dist']:.4f}")

            axes = ['roll', 'pitch', 'yaw', 'tx', 'ty', 'tz']
            stats = {a: np.array([r[a] for r in rows]) for a in axes}
            after_dists = np.array([r['after_dist'] for r in rows])
            rot_errs = np.array([r['rot_err'] for r in rows])
            trans_errs = np.array([r['trans_err'] for r in rows])

            print(f"  === {name} ===")
            # signed mean +/- std separates bias from scatter; MAE alongside for magnitude
            for axis, label in zip(['roll', 'pitch', 'yaw'], rot_lbl):
                v = stats[axis]
                print(f"    {label + ' [deg]':<20s} mean={v.mean():+8.3f}  std={v.std():6.3f}  "
                      f"MAE={np.abs(v).mean():6.3f}")
            for axis, label in zip(['tx', 'ty', 'tz'], trans_lbl):
                v = stats[axis]
                print(f"    {label + ' [' + unit + ']':<20s} mean={v.mean():+8.4f}  std={v.std():6.4f}  "
                      f"MAE={np.abs(v).mean():6.4f}")
            print(f"    (combined rot_error_deg mean={rot_errs.mean():7.3f}, "
                  f"combined trans_error mean={trans_errs.mean():.4f}, "
                  f"after_dist mean={after_dists.mean():.4f})")


if __name__ == '__main__':
    main()

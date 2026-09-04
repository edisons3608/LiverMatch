"""Precompute point clouds for all talus STL meshes in a subject folder and cache
them as .npy files, plus write train/val split lists.

Usage: run this once before training on a new talus dataset directory.
  python prepare_talus_cache.py [talus_dir] [--downsample-factor N] [--cache-name NAME]
"""
import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import open3d as o3d

from lib.talus_acs import invariant_scale

DEFAULT_TALUS_DIR = r"C:\Users\esun3\Documents\talus2\left"
N_POINTS = 8000
TARGET_DIAG = 3.49  # legacy bounding-box-diagonal convention (see stl_to_pcd)
# Median RMS radius the legacy convention produced on this dataset, so switching
# normalisation leaves point spacing -- and every distance threshold tuned
# against it -- unchanged.
TARGET_RMS = 0.9025
VAL_FRACTION = 0.1
SEED = 0
NORM_SIDECAR = "norm.json"


def stl_to_pcd(path, n_points=N_POINTS, seed=0, norm="rms", target_scale=TARGET_RMS,
               target_diag=TARGET_DIAG, return_scale=False):
    """Surface-sample an STL into a centred, rescaled point cloud.

    norm="rms" (default) scales by the RMS radius about the centroid, which is
    rotation-invariant. norm="bbox" is the legacy bounding-box-diagonal scaling;
    that diagonal shifts ~10% with how the bone was posed in the scanner, so it
    injects that much spurious scale variation across subjects (measured spread
    of rms/bbox over 30 bones here: 8.3%). Kept only to rebuild old caches.
    """
    o3d.utility.random.seed(seed)
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(number_of_points=n_points)
    pts = np.asarray(pcd.points)

    if norm == "rms":
        scale = target_scale / invariant_scale(pts)
    elif norm == "bbox":
        scale = target_diag / np.linalg.norm(pts.max(0) - pts.min(0))
    else:
        raise ValueError("norm must be 'rms' or 'bbox', got %r" % (norm,))

    pts = pts * scale
    pts = (pts - pts.mean(0)).astype(np.float32)
    return (pts, 1.0 / scale) if return_scale else pts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('talus_dir', nargs='?', default=DEFAULT_TALUS_DIR)
    parser.add_argument('--downsample-factor', type=float, default=1.0,
                         help='divide the per-bone point count by this factor, e.g. 3 for a 3x lighter cache')
    parser.add_argument('--norm', choices=['rms', 'bbox'], default='rms',
                         help="point-cloud scale normalisation; 'rms' (rotation-invariant, default) "
                              "or 'bbox' (legacy, pose-dependent -- only to rebuild old caches)")
    parser.add_argument('--cache-name', default=None,
                         help='cache subdirectory name; defaults to "cache_pcd" (or "cache_pcd_dsN" when '
                              '--downsample-factor is set) so downsampled caches do not overwrite the full-res one')
    args = parser.parse_args()

    talus_dir = args.talus_dir
    n_points = max(1, int(round(N_POINTS / args.downsample_factor)))
    if args.cache_name is not None:
        cache_name = args.cache_name
    elif args.downsample_factor != 1.0:
        cache_name = f"cache_pcd_ds{args.downsample_factor:g}"
    else:
        cache_name = "cache_pcd"

    cache_dir = os.path.join(talus_dir, cache_name)
    existing = os.path.isdir(cache_dir) and any(f.endswith('.npy') for f in os.listdir(cache_dir))
    os.makedirs(cache_dir, exist_ok=True)

    sidecar_path = os.path.join(cache_dir, NORM_SIDECAR)
    if os.path.exists(sidecar_path):
        with open(sidecar_path) as f:
            prev = json.load(f)
        if prev.get("norm") != args.norm:
            raise SystemExit(f"{cache_dir} was built with norm='{prev.get('norm')}' but --norm={args.norm} "
                             f"was requested. One cache must use one normalisation -- pick a different "
                             f"--cache-name.")
        mm_per_unit = prev.get("mm_per_unit", {})
    elif existing:
        raise SystemExit(f"{cache_dir} holds .npy files but no {NORM_SIDECAR}, so it predates the "
                         f"normalisation change and was built with norm='bbox'. Pass --norm bbox to "
                         f"extend it, or use a different --cache-name for a fresh rms cache.")
    else:
        mm_per_unit = {}
    print(f"n_points per bone: {n_points} (factor {args.downsample_factor:g} of {N_POINTS}), cache dir: {cache_dir}")
    stl_files = sorted(f for f in os.listdir(talus_dir) if f.lower().endswith(".stl"))
    print(f"Found {len(stl_files)} STL files in {talus_dir}")

    cached_names = []
    for i, fname in enumerate(stl_files):
        out_name = os.path.splitext(fname)[0] + ".npy"
        out_path = os.path.join(cache_dir, out_name)
        if not os.path.exists(out_path) or out_name not in mm_per_unit:
            pts, scale = stl_to_pcd(os.path.join(talus_dir, fname), n_points=n_points, seed=i,
                                     norm=args.norm, return_scale=True)
            np.save(out_path, pts)
            mm_per_unit[out_name] = scale
        cached_names.append(out_name)
        if (i + 1) % 25 == 0 or i == len(stl_files) - 1:
            print(f"  processed {i + 1}/{len(stl_files)}")

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(cached_names))
    n_val = max(1, int(len(cached_names) * VAL_FRACTION))
    val_idx = set(perm[:n_val].tolist())

    train_list = [cached_names[i] for i in range(len(cached_names)) if i not in val_idx]
    val_list = [cached_names[i] for i in val_idx]

    with open(os.path.join(cache_dir, "train_list.txt"), "w") as f:
        f.write("\n".join(train_list) + "\n")
    with open(os.path.join(cache_dir, "val_list.txt"), "w") as f:
        f.write("\n".join(val_list) + "\n")

    with open(sidecar_path, "w") as f:
        json.dump(dict(norm=args.norm, target_rms=TARGET_RMS, target_diag=TARGET_DIAG,
                       n_points=n_points, mm_per_unit=mm_per_unit), f, indent=1)

    print(f"Normalisation: {args.norm} -> {sidecar_path} (records mm_per_unit per bone so "
          f"evaluations can report translations in mm)")
    print(f"Cached point clouds: {len(cached_names)} -> {cache_dir}")
    print(f"Train: {len(train_list)}, Val: {len(val_list)}")


if __name__ == "__main__":
    main()

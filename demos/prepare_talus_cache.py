"""Precompute point clouds for all talus STL meshes in a subject folder and cache
them as .npy files, plus write train/val split lists.

Usage: run this once before training on a new talus dataset directory.
"""
import os
import sys
import numpy as np
import open3d as o3d

DEFAULT_TALUS_DIR = r"C:\Users\esun3\Documents\talus2\left"
N_POINTS = 8000
TARGET_DIAG = 3.49  # matches the liver dataset's scaled bounding-box-diagonal convention
VAL_FRACTION = 0.1
SEED = 0


def stl_to_pcd(path, n_points=N_POINTS, target_diag=TARGET_DIAG, seed=0):
    o3d.utility.random.seed(seed)
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(number_of_points=n_points)
    pts = np.asarray(pcd.points)
    diag = np.linalg.norm(pts.max(0) - pts.min(0))
    pts = pts * (target_diag / diag)
    pts = pts - pts.mean(0)
    return pts.astype(np.float32)


def main():
    if len(sys.argv) > 1:
        talus_dir = sys.argv[1]
    else:
        talus_dir = DEFAULT_TALUS_DIR

    cache_dir = os.path.join(talus_dir, "cache_pcd")
    os.makedirs(cache_dir, exist_ok=True)
    stl_files = sorted(f for f in os.listdir(talus_dir) if f.lower().endswith(".stl"))
    print(f"Found {len(stl_files)} STL files in {talus_dir}")

    cached_names = []
    for i, fname in enumerate(stl_files):
        out_name = os.path.splitext(fname)[0] + ".npy"
        out_path = os.path.join(cache_dir, out_name)
        if not os.path.exists(out_path):
            pts = stl_to_pcd(os.path.join(talus_dir, fname), seed=i)
            np.save(out_path, pts)
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

    print(f"Cached point clouds: {len(cached_names)} -> {cache_dir}")
    print(f"Train: {len(train_list)}, Val: {len(val_list)}")


if __name__ == "__main__":
    main()

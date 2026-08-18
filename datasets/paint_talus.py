import os
import sys
[sys.path.append(i) for i in ['.', '..']]
import random

import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class paintTalusDataset(Dataset):
	"""Talus training-pair generator using line-painted partial views.

	Source is the full point cloud. Target is formed by selecting points nearest
	to straight strokes projected onto a locally fitted continuous surface,
	then applying rigid
	transform and noise.
	"""

	def __init__(self, config, mode):
		super(paintTalusDataset, self).__init__()
		self.root_path = config.root_path
		self.mode = mode

		if mode == "train":
			self.file_list = np.loadtxt(config.train_list, dtype=str).tolist()
		else:
			self.file_list = np.loadtxt(config.val_list, dtype=str).tolist()

		self.max_noise = config.max_noise
		self.max_vis = config.max_vis
		self.min_vis = config.min_vis
		self.config = config
		self.paint_min_vis = float(getattr(config, 'paint_min_vis', 0.05))
		self.paint_max_vis = float(getattr(config, 'paint_max_vis', 0.20))

		self.min_lines = int(getattr(config, 'min_lines', 3))
		self.max_lines = int(getattr(config, 'max_lines', 5))
		self.walk_knn = int(getattr(config, 'paint_walk_knn', 12))
		self.min_stroke_points = int(getattr(config, 'paint_min_stroke_points', 70))
		self.max_stroke_points = int(getattr(config, 'paint_max_stroke_points', 220))
		self.max_strokes_factor = float(getattr(config, 'paint_max_strokes_factor', 3.0))
		self.stroke_step_scale = float(getattr(config, 'paint_stroke_step_scale', 0.8))
		self.interline_gap_scale = float(getattr(config, 'paint_interline_gap_scale', 2.2))
		self.min_line_len_scale = float(getattr(config, 'paint_min_line_len_scale', 0.28))
		self.max_line_len_scale = float(getattr(config, 'paint_max_line_len_scale', 0.62))
		self.proj_knn = int(getattr(config, 'paint_proj_knn', 32))
		self.proj_iters = int(getattr(config, 'paint_proj_iters', 2))
		self.use_mesh_projection = bool(getattr(config, 'paint_use_mesh_projection', True))
		self.mesh_root = getattr(config, 'paint_mesh_root', None)
		self.mesh_ext = str(getattr(config, 'paint_mesh_ext', '.stl')).lower()
		self.max_rot_deg = float(getattr(config, 'max_rot_deg', 60.0))
		self.max_trans = float(getattr(config, 'max_trans', 0.3))

		self._mesh_scene_cache = {}

	def __len__(self):
		return len(self.file_list)

	def _resolve_sample_path(self, index):
		if self.mode == "train":
			name = random.choice(self.file_list)
		else:
			name = self.file_list[index % len(self.file_list)]

		if os.path.isabs(name):
			return name
		return os.path.join(self.root_path, name)

	def _build_knn(self, points):
		n_pts = points.shape[0]
		k = max(3, min(self.walk_knn + 1, n_pts))
		tree = cKDTree(points)
		dists, _ = tree.query(points, k=k)
		if len(dists.shape) == 1:
			dists = dists[:, None]
		step_col = min(6, dists.shape[1] - 1)
		local_step = float(np.median(dists[:, step_col])) if step_col >= 1 else 1e-3
		local_step = max(local_step, 1e-4)
		return tree, local_step

	def _resolve_mesh_path(self, sample_path):
		stem = os.path.splitext(os.path.basename(sample_path))[0]

		if self.mesh_root is not None:
			mesh_path = os.path.join(self.mesh_root, stem + self.mesh_ext)
			if os.path.isfile(mesh_path):
				return mesh_path

		sample_dir = os.path.dirname(sample_path)
		mesh_path = os.path.join(sample_dir, stem + self.mesh_ext)
		if os.path.isfile(mesh_path):
			return mesh_path

		return None

	def _get_mesh_scene(self, sample_path):
		mesh_path = self._resolve_mesh_path(sample_path)
		if mesh_path is None:
			return None

		if mesh_path in self._mesh_scene_cache:
			return self._mesh_scene_cache[mesh_path]

		mesh = o3d.io.read_triangle_mesh(mesh_path)
		if mesh is None or len(mesh.triangles) == 0 or len(mesh.vertices) == 0:
			return None

		mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
		scene = o3d.t.geometry.RaycastingScene()
		scene.add_triangles(mesh_t)
		self._mesh_scene_cache[mesh_path] = scene
		return scene

	def _interpolate_polyline(self, polyline, local_step):
		if polyline.shape[0] < 2:
			return np.zeros((0, 3), dtype=np.float32)

		stroke_pts = []
		for i in range(polyline.shape[0] - 1):
			a = polyline[i]
			b = polyline[i + 1]
			seg_len = float(np.linalg.norm(b - a))
			n_step = max(2, int(np.ceil(seg_len / (local_step * self.stroke_step_scale))))
			t = np.linspace(0.0, 1.0, n_step, endpoint=False, dtype=np.float32)
			seg = (1.0 - t[:, None]) * a[None, :] + t[:, None] * b[None, :]
			stroke_pts.append(seg)

		stroke_pts.append(polyline[-1][None, :])
		return np.concatenate(stroke_pts, axis=0).astype(np.float32)

	def _project_points_to_surface(self, query_pts, surface_pts, tree):
		k = max(8, min(self.proj_knn, surface_pts.shape[0]))
		projected = query_pts.astype(np.float32).copy()

		for i in range(projected.shape[0]):
			q = projected[i]
			for _ in range(self.proj_iters):
				_, idx = tree.query(q, k=k)
				if np.isscalar(idx):
					neigh = surface_pts[np.array([idx], dtype=np.int64)]
				else:
					neigh = surface_pts[np.asarray(idx, dtype=np.int64)]

				c = neigh.mean(axis=0)
				x = neigh - c[None, :]
				cov = (x.T @ x) / max(1, x.shape[0])
				_, vecs = np.linalg.eigh(cov)
				n = vecs[:, 0]
				q = q - np.dot(q - c, n) * n

			projected[i] = q.astype(np.float32)

		return projected

	def _project_points_to_mesh_or_surface(self, query_pts, sample_path, surface_pts, tree):
		if self.use_mesh_projection:
			scene = self._get_mesh_scene(sample_path)
			if scene is not None:
				tensor_pts = o3d.core.Tensor(query_pts.astype(np.float32), dtype=o3d.core.Dtype.Float32)
				closest = scene.compute_closest_points(tensor_pts)
				mesh_proj = closest['points'].numpy().astype(np.float32)
				return mesh_proj

		return self._project_points_to_surface(query_pts, surface_pts, tree)

	@staticmethod
	def _smooth_polyline(polyline, n_pass=2):
		if polyline.shape[0] < 3:
			return polyline
		s = polyline.astype(np.float32).copy()
		for _ in range(n_pass):
			s[1:-1] = 0.25 * s[:-2] + 0.5 * s[1:-1] + 0.25 * s[2:]
		return s

	def _sample_projected_stroke(self, points, sample_path, tree, local_step, blocked, bbox_diag):
		n_pts = points.shape[0]
		for _ in range(24):
			anchor_pool = np.nonzero(~blocked)[0]
			if anchor_pool.size == 0:
				anchor_pool = np.arange(n_pts)
			anchor = points[int(np.random.choice(anchor_pool))]

			dir_vec = np.random.normal(size=3).astype(np.float32)
			nrm = np.linalg.norm(dir_vec)
			if nrm < 1e-6:
				continue
			dir_vec /= nrm

			line_len = np.random.uniform(self.min_line_len_scale, self.max_line_len_scale) * bbox_diag
			n_line = random.randint(self.min_stroke_points, self.max_stroke_points)
			t = np.linspace(-0.5 * line_len, 0.5 * line_len, n_line, dtype=np.float32)
			line_pts = anchor[None, :] + t[:, None] * dir_vec[None, :]
			proj_pts = self._project_points_to_mesh_or_surface(line_pts, sample_path, points, tree)
			proj_pts = self._smooth_polyline(proj_pts, n_pass=2)

			proj_idx = tree.query(proj_pts, k=1)[1].astype(np.int64)
			keep = np.ones(proj_idx.shape[0], dtype=bool)
			keep[1:] = proj_idx[1:] != proj_idx[:-1]
			proj_idx = proj_idx[keep]
			polyline = proj_pts[keep]

			if proj_idx.shape[0] < 8:
				continue

			if blocked[proj_idx].mean() > 0.2:
				continue

			interp = self._interpolate_polyline(polyline, local_step)
			if interp.shape[0] < 12:
				continue

			return polyline.astype(np.float32), interp.astype(np.float32), proj_idx

		return None, None, None

	def line_select(self, points, sample_path, p_keep, n_lines=None):
		n_pts = points.shape[0]
		n_keep = max(20, int(round(p_keep * n_pts)))

		if n_lines is None:
			n_lines = random.randint(self.min_lines, self.max_lines)

		tree, local_step = self._build_knn(points)
		selected = np.zeros(n_pts, dtype=bool)
		blocked = np.zeros(n_pts, dtype=bool)
		paint_pts = []
		stroke_polylines = []
		gap_radius = self.interline_gap_scale * local_step
		bbox_diag = float(np.linalg.norm(points.max(0) - points.min(0)))

		accepted_strokes = 0
		attempts = 0
		max_attempts = max(8, int(np.ceil(self.max_strokes_factor * n_lines * 8)))

		while accepted_strokes < n_lines and attempts < max_attempts:
			polyline, interp, proj_idx = self._sample_projected_stroke(points, sample_path, tree, local_step, blocked, bbox_diag)
			if polyline is None:
				attempts += 1
				continue

			stroke_polylines.append(polyline.astype(np.float32))
			paint_pts.append(interp)

			selected[proj_idx] = True
			for idx in proj_idx.tolist():
				near_idx = tree.query_ball_point(points[idx], r=gap_radius)
				if len(near_idx) > 0:
					blocked[np.asarray(near_idx, dtype=np.int64)] = True
			accepted_strokes += 1
			attempts += 1

		if len(paint_pts) == 0:
			return self.line_select(points, sample_path, p_keep, n_lines=n_lines)

		tgt_all = np.concatenate(paint_pts, axis=0).astype(np.float32)
		if tgt_all.shape[0] > n_keep:
			choice = np.random.choice(tgt_all.shape[0], size=n_keep, replace=False)
			tgt_all = tgt_all[choice]
		elif tgt_all.shape[0] < n_keep:
			extra = points[np.random.choice(n_pts, size=n_keep - tgt_all.shape[0], replace=True)]
			tgt_all = np.concatenate([tgt_all, extra.astype(np.float32)], axis=0)

		nn_src = tree.query(tgt_all, k=1)[1].astype(np.int64)
		mask = np.zeros(n_pts, dtype=bool)
		mask[np.unique(nn_src)] = True
		return tgt_all, mask, nn_src, stroke_polylines

	def rand_rigid(self, points):
		max_rad = np.deg2rad(self.max_rot_deg)
		euler = np.random.uniform(-max_rad, max_rad, size=3)
		rot = Rotation.from_euler('zyx', euler).as_matrix().astype(np.float32)
		trans = (np.random.rand(3, 1) * 2 * self.max_trans - self.max_trans).astype(np.float32)
		moved = (np.matmul(rot, points.T) + trans).T
		return moved, rot, trans

	@staticmethod
	def center(points):
		centroid = np.mean(points[:, :3], axis=0)
		return points[:, :3] - centroid

	def get_input_train(self, index, vis=False, p=None, return_meta=False):
		sample_path = self._resolve_sample_path(index)
		src_vs = np.load(sample_path).astype(np.float32)
		src_vs = self.center(src_vs)

		if p is None:
			p = self.paint_min_vis + (self.paint_max_vis - self.paint_min_vis) * np.random.rand(1)[0]
			p = min(p, 1.0)

		tgt_partial, mask, src_map_idx, stroke_src = self.line_select(src_vs, sample_path, p_keep=p)

		sigma = np.random.rand(1)[0] * self.max_noise
		tgt_partial = tgt_partial + (np.random.rand(tgt_partial.shape[0], 3) - 0.5) * sigma

		tgt_pcd, rot, trans = self.rand_rigid(tgt_partial)
		stroke_tgt = [((np.matmul(rot, s.T) + trans).T).astype(np.float32) for s in stroke_src]

		tgt_idx = np.arange(tgt_pcd.shape[0], dtype=np.int64)
		src_idx = src_map_idx[:tgt_idx.shape[0]]
		correspondences = torch.from_numpy(np.stack([src_idx, tgt_idx], axis=1)).long()

		src_pcd = src_vs
		src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
		tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)

		if correspondences.size(0) < 20 or np.unique(src_idx).shape[0] < 20:
			return self.get_input_train(np.random.choice(len(self.file_list), 1)[0], vis=vis, return_meta=return_meta)

		data = (
			src_pcd.astype(np.float32), tgt_pcd.astype(np.float32), src_feats, tgt_feats,
			rot.astype(np.float32), trans.astype(np.float32), correspondences,
			src_pcd.astype(np.float32), tgt_pcd.astype(np.float32), torch.ones(1)
		)

		if not return_meta:
			return data

		meta = {
			'strokes_src': stroke_src,
			'strokes_tgt': stroke_tgt,
		}
		return data, meta

	def __getitem__(self, index, vis=False):
		return self.get_input_train(index, vis=vis)


if __name__ == '__main__':
	from easydict import EasyDict as edict
	from lib.util import load_config

	config_path = "configs/talus.yaml"
	config = load_config(config_path)
	config = edict(config)

	dataset = paintTalusDataset(config, "train")
	data = dataset.__getitem__(0)
	print("src", data[0].shape, "tgt", data[1].shape, "n_corr", data[6].shape)

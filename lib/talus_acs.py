"""Anatomical coordinate system (ACS) for talus registration evaluation.

Why this exists: 6DoF errors reported in the STL/scanner frame are not comparable
across subjects. On this dataset the raw scanner frames differ by ~27 deg (mean;
max 43) because each bone was segmented from a differently-posed whole-body CT,
so a `tx` of 1 mm points in a different anatomical direction for every subject.

The fix here is the standard groupwise-atlas ACS: an anatomical frame is defined
ONCE on a single template bone, then propagated to every other subject by rigid
shape registration (PCA initialisation + ICP). Errors conjugated into that frame
are directly comparable subject-to-subject.

Per-subject PCA is deliberately NOT used as the frame. Measured against a
shape-consistent (ICP) alignment it is off by 6.6 deg on average (max 14.8) on
this dataset, because the talus' 2nd and 3rd principal moments are close
(sqrt-ratios ~0.72 vs ~0.58) so those two axes rotate against each other with
shape. PCA is used only as an ICP initialiser.

When a statistical shape model covering the bones is available, prefer
lib/talus_ssm.py: its corresponded clouds give the same frame without any
cross-shape registration, which removes ~1.7 deg (max 2.5) of per-subject frame
inconsistency relative to the template route here. This module stays the
fallback for bones the SSM does not contain.
"""
import itertools
import json
import os

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

# Axis slot names. The frame always has 3 ordered axes; what they are CALLED
# depends on how the template frame was defined.
PCA_AXIS_NAMES = ("a1", "a2", "a3")
PCA_ROT_NAMES = ("rot_a1", "rot_a2", "rot_a3")
# Clinical talus convention, available once landmarks are supplied:
#   ML = medial-lateral  (trochlear cylinder axis)
#   AP = anterior-posterior (body -> head/neck)
#   SI = superior-inferior (ML x AP)
CLINICAL_AXIS_NAMES = ("ML", "AP", "SI")
CLINICAL_ROT_NAMES = ("dorsi/plantarflex", "inv/eversion", "int/ext_rot")


# --------------------------------------------------------------------------
# scale / basic geometry
# --------------------------------------------------------------------------

def invariant_scale(points):
    """RMS radius about the centroid -- a rotation-invariant size measure.

    Use this instead of the bounding-box diagonal. The bbox diagonal varies by
    ~10% with pose on a single fixed talus (measured: mean 87.79, std 1.60 over
    200 random orientations), so normalising by it injects spurious scale
    variation across subjects that lands straight in the translation errors.
    The RMS radius is exactly invariant (std 0.0 on the same test).
    """
    pts = np.asarray(points, dtype=np.float64)[:, :3]
    return float(np.sqrt(((pts - pts.mean(0)) ** 2).sum(1).mean()))


def normalize(points):
    """Centre at the centroid and scale to unit RMS radius.

    Returns (unit_points, centroid, scale) with unit_points = (points - c) / s.
    """
    pts = np.asarray(points, dtype=np.float64)[:, :3]
    c = pts.mean(0)
    s = invariant_scale(pts)
    return (pts - c) / s, c, s


def pca_axes(points):
    """Principal axes of the point cloud, columns ordered by descending moment.

    Signs are disambiguated by third moment (skewness) along each axis, which is
    a shape-derived choice rather than an arbitrary eigensolver one, so repeated
    calls on resampled versions of the same bone agree. The result is
    right-handed (det = +1).
    """
    pts = np.asarray(points, dtype=np.float64)[:, :3]
    X = pts - pts.mean(0)
    cov = X.T @ X / len(X)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]

    proj = X @ V
    skew = (proj ** 3).mean(0)
    for k in range(3):
        # fall back to "largest-magnitude coordinate is positive" when the bone is
        # near-symmetric along this axis and the skew sign is not meaningful
        ref = skew[k] if abs(skew[k]) > 1e-9 else V[np.argmax(np.abs(V[:, k])), k]
        if ref < 0:
            V[:, k] *= -1
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1
    return pts.mean(0), V, w


def _pcd(points):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return p


# --------------------------------------------------------------------------
# rigid registration used to propagate the frame
# --------------------------------------------------------------------------

def _init_candidates(V_src, V_tgt):
    """Rotations aligning the source PCA frame onto the template PCA frame.

    Enumerates sign flips and the 2<->3 axis swap. The swap is included because
    the talus' 2nd/3rd moments are close enough that their order is not reliable
    across subjects; only right-handed candidates are kept.
    """
    cands = []
    for perm in ((0, 1, 2), (0, 2, 1)):
        Vp = V_src[:, perm]
        for signs in itertools.product((1, -1), repeat=3):
            Vs = Vp * np.array(signs)
            if np.linalg.det(Vs) < 0:
                continue
            R = V_tgt @ Vs.T
            T = np.eye(4)
            T[:3, :3] = R
            cands.append(T)
    return cands


def register_to_template(src_unit, tgt_unit, max_corr=0.25, max_iter=200):
    """Rigid-register a unit-normalised bone onto a unit-normalised template.

    Multi-start ICP over the PCA-frame candidates; returns the best transform
    (4x4, maps src_unit into tgt_unit coordinates) plus its fitness and RMSE.
    """
    _, V_src, _ = pca_axes(src_unit)
    _, V_tgt, _ = pca_axes(tgt_unit)
    src_pcd, tgt_pcd = _pcd(src_unit), _pcd(tgt_unit)
    estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)

    best, best_score = None, np.inf
    for T0 in _init_candidates(V_src, V_tgt):
        res = o3d.pipelines.registration.registration_icp(
            src_pcd, tgt_pcd, max_corr, T0, estimator, criteria)
        score = res.inlier_rmse / max(res.fitness, 1e-6)
        if score < best_score:
            best, best_score = res, score
    return np.asarray(best.transformation), float(best.fitness), float(best.inlier_rmse)


# --------------------------------------------------------------------------
# the frame itself
# --------------------------------------------------------------------------

class AnatomicalFrame(object):
    """An orthonormal frame attached to one bone.

    R       -- 3x3, COLUMNS are the anatomical axes expressed in the coordinates
               of the point cloud this frame belongs to (det = +1).
    origin  -- anatomical origin, same coordinates.
    mm_per_unit -- multiply translations in those coordinates by this to get mm;
               None when the mapping to mm is unknown.
    """

    def __init__(self, R, origin, mm_per_unit=None, axis_names=PCA_AXIS_NAMES,
                 rot_names=PCA_ROT_NAMES, source="pca", fitness=None, rmse=None,
                 ref_centroid=None, ref_scale=None):
        self.R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        self.origin = np.asarray(origin, dtype=np.float64).reshape(3)
        self.mm_per_unit = None if mm_per_unit is None else float(mm_per_unit)
        self.axis_names = tuple(axis_names)
        self.rot_names = tuple(rot_names)
        self.source = source
        self.fitness = fitness
        self.rmse = rmse
        # centroid and RMS size of the cloud this frame was built for; a cached
        # frame is only valid for a cloud normalised the same way
        self.ref_centroid = None if ref_centroid is None else np.asarray(ref_centroid, dtype=np.float64)
        self.ref_scale = None if ref_scale is None else float(ref_scale)

    def matches(self, points, rtol=1e-3):
        """Whether this frame was built for a cloud like `points`."""
        if self.ref_centroid is None or self.ref_scale is None:
            return False
        pts = np.asarray(points, dtype=np.float64)[:, :3]
        scale = invariant_scale(pts)
        return (abs(scale - self.ref_scale) <= rtol * max(self.ref_scale, 1e-12)
                and np.allclose(pts.mean(0), self.ref_centroid, rtol=0, atol=rtol * self.ref_scale))

    def rescaled(self, centroid, scale, mm_per_unit=None):
        """Map this frame out of unit-normalised coordinates into (c + s * x)."""
        return AnatomicalFrame(self.R, np.asarray(centroid) + scale * self.origin,
                               mm_per_unit if mm_per_unit is not None else self.mm_per_unit,
                               self.axis_names, self.rot_names, self.source,
                               self.fitness, self.rmse, centroid, scale)

    def to_dict(self):
        return dict(R=self.R.tolist(), origin=self.origin.tolist(),
                    mm_per_unit=self.mm_per_unit, axis_names=list(self.axis_names),
                    rot_names=list(self.rot_names), source=self.source,
                    fitness=self.fitness, rmse=self.rmse,
                    ref_centroid=None if self.ref_centroid is None else self.ref_centroid.tolist(),
                    ref_scale=self.ref_scale)

    @classmethod
    def from_dict(cls, d):
        return cls(d["R"], d["origin"], d.get("mm_per_unit"),
                   d.get("axis_names", PCA_AXIS_NAMES), d.get("rot_names", PCA_ROT_NAMES),
                   d.get("source", "pca"), d.get("fitness"), d.get("rmse"),
                   d.get("ref_centroid"), d.get("ref_scale"))


def template_frame_from_pca(template_unit):
    """Fallback template frame: the template's own principal axes.

    Fully adequate for cross-subject COMPARABILITY -- every subject inherits this
    same frame through registration, so the axes mean the same thing everywhere.
    What it does not give you is clinical INTERPRETABILITY: a1/a2/a3 are moment
    axes, not dorsiflexion/inversion/rotation. Use
    `template_frame_from_landmarks` for that.
    """
    origin, V, _ = pca_axes(template_unit)
    return AnatomicalFrame(V, origin, axis_names=PCA_AXIS_NAMES,
                           rot_names=PCA_ROT_NAMES, source="template-pca")


def template_frame_from_landmarks(template_unit, trochlea_axis, head_point, origin=None):
    """Clinical talus frame, defined once on the template.

    trochlea_axis -- direction of the trochlear cylinder axis (medial-lateral),
                     e.g. the axis of a cylinder fitted to the trochlear dome.
                     Point it from medial to lateral.
    head_point    -- a point on the talar head/neck; body-centroid -> head gives
                     the anterior direction.
    origin        -- anatomical origin; defaults to the centroid.

    All arguments are in the template's unit-normalised coordinates. Axes come
    out as columns [ML, AP, SI], so rotations about them read as
    dorsi/plantarflexion, inversion/eversion and internal/external rotation.
    """
    pts = np.asarray(template_unit, dtype=np.float64)[:, :3]
    e_ml = np.asarray(trochlea_axis, dtype=np.float64)
    e_ml = e_ml / np.linalg.norm(e_ml)

    ap = np.asarray(head_point, dtype=np.float64) - pts.mean(0)
    ap = ap - (ap @ e_ml) * e_ml          # orthogonalise against ML
    e_ap = ap / np.linalg.norm(ap)

    e_si = np.cross(e_ml, e_ap)
    V = np.column_stack([e_ml, e_ap, e_si])
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1
    return AnatomicalFrame(V, pts.mean(0) if origin is None else origin,
                           axis_names=CLINICAL_AXIS_NAMES, rot_names=CLINICAL_ROT_NAMES,
                           source="template-landmarks")


def propagate_frame(subject_points, template_unit, template_frame, mm_per_unit=None,
                    max_corr=0.25, max_iter=200):
    """Carry the template's anatomical frame onto one subject.

    `subject_points` is whatever point array the evaluation actually uses (it may
    already be centred/rescaled by the loader); the returned frame is expressed
    in THOSE coordinates, so it can be applied without further bookkeeping.
    """
    src_unit, c_s, s_s = normalize(subject_points)
    T, fitness, rmse = register_to_template(src_unit, template_unit, max_corr, max_iter)
    R_T, t_T = T[:3, :3], T[:3, 3]

    # invert the registration to bring the template's frame back to the subject
    R_A = R_T.T @ template_frame.R
    o_A = R_T.T @ (template_frame.origin - t_T)

    frame = AnatomicalFrame(R_A, o_A, None, template_frame.axis_names,
                            template_frame.rot_names, template_frame.source, fitness, rmse)
    return frame.rescaled(c_s, s_s, mm_per_unit)


# --------------------------------------------------------------------------
# the error metric
# --------------------------------------------------------------------------

def error_6dof(tsfm_pred, rot_gt, trans_gt, frame=None, signed=True, degrees=True):
    """Per-axis 6DoF registration error, optionally in an anatomical frame.

    tsfm_pred -- 4x4 predicted source->target transform.
    rot_gt, trans_gt -- ground-truth source->target rotation / translation.
    frame -- AnatomicalFrame in SOURCE coordinates. When given, rotations are
             conjugated into that frame and the translation is the displacement
             OF THE ANATOMICAL ORIGIN, expressed on the anatomical axes, in mm
             if the frame carries mm_per_unit. When None the error is reported on
             world axes at the world origin (the old, non-comparable behaviour).

    Returns (r1, r2, r3, t1, t2, t3). Signed by default: a consistent bias and
    symmetric scatter are different failures and abs() collapses them into the
    same number. Aggregate signed mean +/- SD, and take abs() afterwards if you
    also want an MAE.

    Frame maths, for the record. The error transform in source coordinates is
    T_gt^-1 . T_pred, so with R_A the anatomical axes and o_A the anatomical
    origin (both in source coordinates):

        R_err = R_A^T (R_gt^T R_pred) R_A
        d     = (R_pred o_A + t_pred) - (R_gt o_A + t_gt)
        t_err = R_A^T R_gt^T d

    The conjugation is what makes rotations comparable across subjects; without
    it Euler angles are still read off world axes. Evaluating the displacement at
    o_A rather than differencing t_pred - t_gt matters because the latter is the
    displacement at the world origin, so rotation error leaks into translation
    through the lever arm to the bone.
    """
    tsfm_pred = np.asarray(tsfm_pred, dtype=np.float64)
    R_pred, t_pred = tsfm_pred[:3, :3], tsfm_pred[:3, 3]
    R_gt = np.asarray(rot_gt, dtype=np.float64).reshape(3, 3)
    t_gt = np.asarray(trans_gt, dtype=np.float64).reshape(3)

    if frame is None:
        R_err = R_pred @ R_gt.T
        t_err = t_pred - t_gt
    else:
        R_A, o_A = frame.R, frame.origin
        R_err = R_A.T @ (R_gt.T @ R_pred) @ R_A
        d = (R_pred @ o_A + t_pred) - (R_gt @ o_A + t_gt)
        t_err = R_A.T @ R_gt.T @ d
        if frame.mm_per_unit is not None:
            t_err = t_err * frame.mm_per_unit

    r1, r2, r3 = Rotation.from_matrix(R_err).as_euler('xyz', degrees=degrees)
    t1, t2, t3 = t_err
    if not signed:
        return abs(r1), abs(r2), abs(r3), abs(t1), abs(t2), abs(t3)
    return r1, r2, r3, t1, t2, t3


def axis_labels(frame=None):
    """Column headers matching `error_6dof`'s return order."""
    if frame is None:
        return ("roll", "pitch", "yaw", "tx", "ty", "tz")
    return tuple("rot_" + n for n in frame.axis_names) + tuple("t_" + n for n in frame.axis_names)


# --------------------------------------------------------------------------
# template selection + on-disk cache
# --------------------------------------------------------------------------

def choose_template(point_clouds, names=None, verbose=True):
    """Pick the medoid bone: the one with the lowest mean registration cost to
    all the others. A template close to the population mean keeps every
    subject's propagation well-conditioned."""
    units = [normalize(p)[0] for p in point_clouds]
    n = len(units)
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            _, fit, rmse = register_to_template(units[i], units[j])
            cost[i, j] = rmse / max(fit, 1e-6)
    mean_cost = cost.sum(1) / (n - 1)
    best = int(np.argmin(mean_cost))
    if verbose:
        for i in np.argsort(mean_cost):
            tag = "  <-- template" if i == best else ""
            label = names[i] if names else "#%d" % i
            print("  %s  mean_cost=%.5f%s" % (label, mean_cost[i], tag))
    return best, mean_cost


class ACSLibrary(object):
    """Template frame + per-subject frames, cached to a .json on disk.

    Registration is the expensive part (multi-start ICP per subject), so frames
    are computed once and reused across evaluation runs.
    """

    def __init__(self, template_points, template_frame=None, cache_path=None):
        self.template_unit, self.template_centroid, self.template_scale = normalize(template_points)
        self.frame = template_frame or template_frame_from_pca(self.template_unit)
        self.cache_path = cache_path
        self._cache = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                blob = json.load(f)
            if blob.get("source") == self.frame.source:
                self._cache = {k: AnatomicalFrame.from_dict(v) for k, v in blob["frames"].items()}

    def frame_for(self, key, subject_points, mm_per_unit=None):
        cached = self._cache.get(key)
        # a frame is expressed in the coordinates of the cloud it was built for, so a
        # cache entry made under a different normalisation must not be reused
        if cached is not None and cached.matches(subject_points):
            if mm_per_unit is not None and cached.mm_per_unit is None:
                cached.mm_per_unit = float(mm_per_unit)
            return cached
        frame = propagate_frame(subject_points, self.template_unit, self.frame, mm_per_unit)
        self._cache[key] = frame
        return frame

    def save(self):
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(dict(source=self.frame.source,
                           frames={k: v.to_dict() for k, v in self._cache.items()}), f, indent=1)

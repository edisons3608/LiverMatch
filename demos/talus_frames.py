"""Shared anatomical-frame setup for the talus evaluation demos.

Every eval script pulls its frames from here so they all report on the SAME
anatomical axes -- numbers from different scripts are then directly comparable.

Frames come from lib/talus_acs.py: a frame is defined once on one medoid bone
(the bone with the lowest mean registration cost to a sample of the rest -- see
lib.talus_acs.choose_template) and carried to every other subject by rigid
shape registration (PCA-initialised, multi-start ICP).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import open3d as o3d

from lib.talus_acs import ACSLibrary, invariant_scale

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Medoid of an 18-bone sample of talus2/left. Re-derive with
# lib.talus_acs.choose_template if the dataset changes.
TEMPLATE_STL = r"C:\Users\esun3\Documents\talus2\left\200129-xx-m-060xxxxxx-tal-l-c-d-s%.stl"
TEMPLATE_N_POINTS = 8000          # fixed, so the template frame never depends on eval density
TEMPLATE_SEED = 0
ACS_CACHE = os.path.join(REPO_ROOT, "demos", "talus_acs_cache.json")

_library = None


def _sample(path, n_points, seed):
    o3d.utility.random.seed(seed)
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.vertices) == 0:
        raise FileNotFoundError("could not read a mesh from %s" % path)
    mesh.compute_vertex_normals()
    return np.asarray(mesh.sample_points_poisson_disk(number_of_points=n_points).points)


def get_library():
    """The template library, built once per process and cached to disk across runs."""
    global _library
    if _library is None:
        tpl = _sample(TEMPLATE_STL, TEMPLATE_N_POINTS, TEMPLATE_SEED)
        _library = ACSLibrary(tpl, cache_path=ACS_CACHE)
    return _library


def mm_per_unit_from_cache(sample_path):
    """mm-per-unit for a cached .npy, from the norm.json prepare_talus_cache writes.

    Returns None for caches built before that sidecar existed, in which case
    translations stay in normalised units rather than being silently wrong.
    """
    sidecar = os.path.join(os.path.dirname(sample_path), 'norm.json')
    if not os.path.exists(sidecar):
        return None
    with open(sidecar) as f:
        import json
        blob = json.load(f)
    return blob.get('mm_per_unit', {}).get(os.path.basename(sample_path))


def frame_for(key, full_points, mm_per_unit=None, save=True):
    """Anatomical frame for one bone, in the coordinates of `full_points`.

    `full_points` MUST be the complete bone, never a partial/cropped view: the
    frame comes from registering the whole shape, and on a 60% crop that
    registration can land in a completely wrong pose (measured drifts of 99 and
    178 deg). In the same-bone partial-view scenarios the source cloud is the
    full bone, so pass that one.
    """
    lib = get_library()
    frame = lib.frame_for(key, full_points, mm_per_unit)
    if save:
        lib.save()
    return frame


def frame_for_stl(path, full_points, n_points=None, mm_per_unit=None, save=True):
    """Frame for a bone identified by its STL path.

    When `mm_per_unit` is not given it is recovered from the mesh itself, so
    translations can be reported in millimetres regardless of how the caller
    normalised its point cloud. The cache key includes `n_points` because the
    frame depends on the evaluation cloud that was registered.
    """
    key = os.path.basename(path)
    if n_points is not None:
        key = "%s@%d" % (key, n_points)
    if mm_per_unit is None:
        mm = invariant_scale(_sample(path, TEMPLATE_N_POINTS, TEMPLATE_SEED))
        mm_per_unit = mm / invariant_scale(full_points)
    return frame_for(key, full_points, mm_per_unit, save)

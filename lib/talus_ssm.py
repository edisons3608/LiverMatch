"""Anatomical frames taken from a talus statistical shape model.

This is the preferred source of frames when the SSM covers the bones being
evaluated, and it beats the template-registration route in lib/talus_acs.py on
both counts that matter:

* The SSM's corresponded point clouds are already groupwise (Procrustes)
  aligned -- the rotation fitting any subject's corresponded cloud onto the mean
  shape is exactly 0 deg -- so the model's own space IS a consistent anatomical
  frame, established over the whole population rather than against one bone.

* Getting that frame onto a subject only needs the subject's OWN corresponded
  cloud registered to its evaluation point cloud. That is the same bone in two
  representations, so the fit is near-exact (fitness 1.000, rmse ~0.030, which
  is just the sampling difference between 6002 model vertices and the sampled
  cloud). Registering one bone to a different template bone instead carries
  rmse 0.049-0.074, and the two routes end up 0.9-2.0 deg apart -- frame error
  that using correspondence simply removes.

The frame's axes are the principal axes of the MEAN shape, so they are defined
by average anatomy rather than by any one subject. They are consistent across
every subject, which is what makes per-axis errors comparable; for clinically
named axes (dorsiflexion / inversion / internal rotation) pass a frame built by
lib.talus_acs.template_frame_from_landmarks instead.
"""
import os

from lib.talus_acs import (AnatomicalFrame, normalize, register_to_template,
                           template_frame_from_pca)


class SSMFrames(object):
    """Per-subject anatomical frames read out of an SSM .h5.

    Expects the layout written by the talus SSM builder:
      provenance/file_names     (n_subjects,) source mesh paths
      model/corresponded_pc     (n_subjects, 3 * n_points) groupwise-aligned clouds
      model/mean_shape          (3 * n_points,)
    """

    def __init__(self, h5_path, frame=None):
        import h5py                      # only needed when an SSM is actually used

        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            raw_names = f["provenance/file_names"][()]
            self.names = [os.path.basename(n.decode() if isinstance(n, bytes) else str(n))
                          for n in raw_names]
            self.corresponded = f["model/corresponded_pc"][()].reshape(len(self.names), -1, 3)
            self.mean_shape = f["model/mean_shape"][()].reshape(-1, 3)
        # indexed by full basename and by stem, so callers holding a cache name
        # ("200001-....npy") or a bare stem resolve the same as an .stl path
        self._index = {}
        for i, n in enumerate(self.names):
            self._index[n] = i
            self._index[os.path.splitext(n)[0]] = i

        self.mean_unit, _, _ = normalize(self.mean_shape)
        self.frame = frame or template_frame_from_pca(self.mean_unit)
        self.frame.source = "ssm-mean-pca" if frame is None else self.frame.source

    @staticmethod
    def _key(name):
        return os.path.splitext(os.path.basename(name))[0]

    def __contains__(self, basename):
        return self._key(basename) in self._index

    def frame_for(self, basename, eval_points, mm_per_unit=None):
        """Frame for one subject, in the coordinates of `eval_points`.

        `eval_points` must be the FULL bone -- the registration matches whole
        shapes, and a partial view can settle in a wholly wrong pose.
        """
        i = self._index[self._key(basename)]
        own_unit, _, _ = normalize(self.corresponded[i])
        sub_unit, c_s, s_s = normalize(eval_points)

        # same bone, two representations -- this fit is near-exact
        T, fitness, rmse = register_to_template(own_unit, sub_unit)

        R_A = T[:3, :3] @ self.frame.R
        o_A = T[:3, :3] @ self.frame.origin + T[:3, 3]
        frame = AnatomicalFrame(R_A, o_A, None, self.frame.axis_names,
                                self.frame.rot_names, self.frame.source, fitness, rmse)
        return frame.rescaled(c_s, s_s, mm_per_unit)

    def mean_shape_mm(self, norm_scale=None):
        """Mean shape in millimetres, if the model's norm_scale is known."""
        if norm_scale is None:
            import h5py
            with h5py.File(self.h5_path, "r") as f:
                norm_scale = f.attrs.get("norm_scale")
        if norm_scale is None:
            return None
        return self.mean_shape / float(norm_scale)

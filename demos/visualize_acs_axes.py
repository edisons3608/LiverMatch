"""Plot each subject's talus with its propagated anatomical (ACS) axes drawn on it,
so the frame that error_6dof reports against can be checked by eye.

Axes are colour-coded consistently across subjects (a1=red, a2=green, a3=blue);
if they land in visibly different anatomical spots per bone, the frame is not
propagating correctly. The template bone gets its own panel for reference.
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from talus_demo import stl_to_pcd
from talus_frames import frame_for_stl, TEMPLATE_STL, get_library
from lib.talus_acs import invariant_scale

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TALUS_DIR = r"C:\Users\esun3\Documents\talus_small"
AXIS_COLORS = ("crimson", "seagreen", "royalblue")


def axis_traces(frame, length, show_legend):
    traces = []
    for k in range(3):
        d = frame.R[:, k] * length
        p0, p1 = frame.origin, frame.origin + d
        traces.append(go.Scatter3d(
            x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
            mode="lines+text",
            line=dict(color=AXIS_COLORS[k], width=10),
            text=["", frame.axis_names[k]],
            textposition="top center",
            textfont=dict(color=AXIS_COLORS[k], size=13),
            name=frame.axis_names[k],
            legendgroup=frame.axis_names[k],
            showlegend=show_legend,
        ))
    traces.append(go.Scatter3d(
        x=[frame.origin[0]], y=[frame.origin[1]], z=[frame.origin[2]],
        mode="markers", marker=dict(color="black", size=4),
        name="origin", legendgroup="origin", showlegend=show_legend,
    ))
    return traces


def panel(fig, col, title_extra, pts, frame, show_legend):
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers", marker=dict(size=1.5, color="lightgray", opacity=0.5),
        name="bone surface", legendgroup="bone", showlegend=show_legend,
    ), row=1, col=col)

    length = invariant_scale(pts) * 1.6
    for tr in axis_traces(frame, length, show_legend):
        fig.add_trace(tr, row=1, col=col)

    fig.layout.annotations[col - 1].text = title_extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-subjects", type=int, default=4,
                         help="how many talus2/left/talus_small subjects to plot, in addition "
                              "to the template panel")
    parser.add_argument("--n-points", type=int, default=4000)
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "demos", "talus_acs_axes.html"))
    args = parser.parse_args()

    files = sorted(f for f in os.listdir(TALUS_DIR) if f.lower().endswith(".stl"))[:args.n_subjects]
    n_panels = 1 + len(files)

    fig = make_subplots(
        rows=1, cols=n_panels,
        specs=[[{"type": "scene"}] * n_panels],
        subplot_titles=["template"] + [f[:6] for f in files],
    )

    # template panel: the frame is native to it, origin/axes come straight off it,
    # no registration involved
    lib = get_library()
    tpl_pts = lib.template_unit * lib.template_scale + lib.template_centroid
    panel(fig, 1, f"template ({os.path.basename(TEMPLATE_STL)[:6]})", tpl_pts, lib.frame,
          show_legend=True)

    for i, fname in enumerate(files):
        path = os.path.join(TALUS_DIR, fname)
        pts, mm_per_unit = stl_to_pcd(path, n_points=args.n_points, seed=0, return_scale=True)
        frame = frame_for_stl(path, pts, n_points=args.n_points, mm_per_unit=mm_per_unit)
        panel(fig, i + 2, f"{fname[:6]}  fit={frame.fitness:.3f} rmse={frame.rmse:.4f}",
              pts, frame, show_legend=False)
        print(f"{fname[:6]}: fitness={frame.fitness:.3f} rmse={frame.rmse:.4f} "
              f"origin={np.round(frame.origin, 3)}")

    scene_kwargs = dict(aspectmode="data", camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)))
    layout_scenes = {f"scene{'' if k == 0 else k + 1}": scene_kwargs for k in range(n_panels)}
    fig.update_layout(
        title="ACS anatomical axes per subject (a1=red, a2=green, a3=blue) -- "
              "same frame propagated from the template by registration",
        legend=dict(orientation="h", y=-0.05),
        width=380 * n_panels, height=600,
        margin=dict(l=10, r=10, t=90, b=10),
        **layout_scenes,
    )
    fig.write_html(args.out, include_plotlyjs="cdn")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()

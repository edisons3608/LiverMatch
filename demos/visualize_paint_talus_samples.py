import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from easydict import EasyDict as edict

from lib.util import load_config
from datasets.paint_talus import paintTalusDataset


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "talus.yaml")
OUT_HTML = os.path.join(REPO_ROOT, "demos", "paint_talus_samples.html")
OUT_DIR = os.path.join(REPO_ROOT, "demos", "paint_talus_sample_viz")
N_SAMPLES = 4


def point_trace(xyz, color, name, size=2, opacity=0.8, show_legend=True):
    return go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode="markers",
        marker=dict(size=size, color=color, opacity=opacity),
        name=name,
        legendgroup=name,
        showlegend=show_legend,
    )


def line_trace(xyz, color, name, width=5, opacity=0.95, show_legend=True):
    return go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode="lines",
        connectgaps=False,
        line=dict(width=width, color=color),
        opacity=opacity,
        name=name,
        legendgroup=name,
        showlegend=show_legend,
    )


def main():
    config = edict(load_config(CONFIG_PATH))
    dataset = paintTalusDataset(config, "train")
    os.makedirs(OUT_DIR, exist_ok=True)

    fig = make_subplots(
        rows=N_SAMPLES,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}] for _ in range(N_SAMPLES)],
        subplot_titles=sum([
            [f"Sample {i + 1}: source + painted subset", f"Sample {i + 1}: transformed noisy target"]
            for i in range(N_SAMPLES)
        ], []),
        horizontal_spacing=0.03,
        vertical_spacing=0.04,
    )

    for i in range(N_SAMPLES):
        sample, meta = dataset.get_input_train(i, return_meta=True)
        src_pcd, tgt_pcd, _, _, _, _, correspondences, _, _, _ = sample

        src_idx = correspondences[:, 0].cpu().numpy()
        src_painted = src_pcd[src_idx]
        strokes_src = meta["strokes_src"]
        strokes_tgt = meta["strokes_tgt"]

        show_legend = i == 0

        fig.add_trace(
            point_trace(src_pcd, "lightgray", "source full", size=2, opacity=0.35, show_legend=show_legend),
            row=i + 1,
            col=1,
        )
        fig.add_trace(
            point_trace(src_painted, "crimson", "painted subset", size=3, opacity=0.9, show_legend=show_legend),
            row=i + 1,
            col=1,
        )
        for s in strokes_src:
            fig.add_trace(
                line_trace(s, "black", "painted line", width=6, opacity=0.95, show_legend=False),
                row=i + 1,
                col=1,
            )

        fig.add_trace(
            point_trace(tgt_pcd, "royalblue", "target transformed+noise", size=2, opacity=0.9, show_legend=show_legend),
            row=i + 1,
            col=2,
        )
        for s in strokes_tgt:
            fig.add_trace(
                line_trace(s, "darkorange", "transformed line", width=6, opacity=0.95, show_legend=False),
                row=i + 1,
                col=2,
            )

        one = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "scene"}, {"type": "scene"}]],
            subplot_titles=("source + painted lines", "transformed noisy target lines"),
            horizontal_spacing=0.03,
        )
        one.add_trace(point_trace(src_pcd, "lightgray", "source full", size=2, opacity=0.35), row=1, col=1)
        one.add_trace(point_trace(src_painted, "crimson", "painted subset", size=3, opacity=0.9), row=1, col=1)
        one.add_trace(point_trace(tgt_pcd, "royalblue", "target transformed+noise", size=2, opacity=0.9), row=1, col=2)
        for s in strokes_src:
            one.add_trace(line_trace(s, "black", "painted line", width=6, opacity=0.95), row=1, col=1)
        for s in strokes_tgt:
            one.add_trace(line_trace(s, "darkorange", "transformed line", width=6, opacity=0.95), row=1, col=2)
        one.update_layout(
            title=f"paint_talus sample {i + 1}",
            width=1200,
            height=520,
            legend=dict(orientation="h", y=1.04),
            margin=dict(l=10, r=10, t=70, b=10),
        )
        one.layout.scene.update(aspectmode="data")
        one.layout.scene2.update(aspectmode="data")
        one_out = os.path.join(OUT_DIR, f"sample_{i + 1}.html")
        one.write_html(one_out, include_plotlyjs="cdn")

    for i in range(N_SAMPLES):
        scene_left = f"scene{2 * i + 1}" if i > 0 else "scene"
        scene_right = f"scene{2 * i + 2}"
        fig.layout[scene_left].update(aspectmode="data")
        fig.layout[scene_right].update(aspectmode="data")

    fig.update_layout(
        title="paint_talus samples: line-painted subset -> rigid transform + noise",
        width=1400,
        height=350 * N_SAMPLES,
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=10, r=10, t=70, b=10),
    )

    fig.write_html(OUT_HTML, include_plotlyjs="cdn")
    print(f"Saved visualization to {OUT_HTML}")
    print(f"Saved per-sample visualizations to {OUT_DIR}")


if __name__ == "__main__":
    main()

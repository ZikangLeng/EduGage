from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


DEFAULT_RUN_ID = "20260429T224314Z__worker_gpu2_slot2__trial_004"
DEFAULT_SELECTIONS = (
    (16, "radiative_transfer"),
    (16, "ideal_rocket_equation"),
    (11, "xray_diffraction"),
    (13, "atmosphere_pressure_composition"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot predicted vs reported scores over time for selected participant-video traces."
    )
    parser.add_argument(
        "--artifact-root",
        default=str(Path(__file__).resolve().parents[1] / "artifacts"),
        help="Artifact root containing runs/ and windows/.",
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help="Run ID to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory. Defaults under artifacts/plots/<run_id>/.",
    )
    parser.add_argument(
        "--selections",
        nargs="*",
        default=None,
        help="Optional selections in the form participant_id:video_uid .",
    )
    parser.add_argument(
        "--prediction-mode",
        choices=("continuous", "rounded"),
        default="continuous",
        help="Whether to plot the continuous regression score or the rounded 1-5 prediction.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Use presentation-style formatting for cleaner standalone figures.",
    )
    parser.add_argument(
        "--title-participant-override",
        type=int,
        default=None,
        help="Optional participant id to show in the title without changing the selected trace.",
    )
    return parser.parse_args()


def _resolve_selections(raw: list[str] | None) -> list[tuple[int, str]]:
    if not raw:
        return list(DEFAULT_SELECTIONS)
    selections: list[tuple[int, str]] = []
    for token in raw:
        participant_text, video_uid = token.split(":", maxsplit=1)
        selections.append((int(participant_text), str(video_uid)))
    return selections


def _load_trace_data(artifact_root: Path, run_id: str) -> pd.DataFrame:
    predictions_path = artifact_root / "runs" / run_id / "phase_e" / "predictions.parquet"
    windows_path = artifact_root / "windows" / "window_index.parquet"
    predictions_df = pd.read_parquet(predictions_path)
    windows_df = pd.read_parquet(windows_path)[
        ["window_id", "t_start_video_sec", "t_end_video_sec"]
    ]
    merged = predictions_df.merge(windows_df, on="window_id", how="left")
    merged["time_center_sec"] = 0.5 * (
        merged["t_start_video_sec"].astype(float) + merged["t_end_video_sec"].astype(float)
    )
    merged["time_center_min"] = merged["time_center_sec"] / 60.0
    # regression_score is stored on the 0--4 scale; shift it back to the 1--5 label scale.
    merged["predicted_score"] = merged["regression_score"].astype(float) + 1.0
    merged["predicted_score_rounded"] = merged["predicted_score"].round().clip(
        lower=1.0,
        upper=5.0,
    )
    merged["reported_score"] = merged["y_true"].astype(float)
    return merged


def _prediction_column_and_label(prediction_mode: str) -> tuple[str, str]:
    if prediction_mode == "rounded":
        return "predicted_score_rounded", "Predicted score (rounded)"
    return "predicted_score", "Predicted score"


def _plot_styles(pretty: bool) -> dict[str, object]:
    if pretty:
        return {
            "reported_color": "#0f4c81",
            "predicted_color": "#d95f02",
            "reported_marker": "o",
            "predicted_marker": "D",
            "reported_linewidth": 2.8,
            "predicted_linewidth": 2.8,
            "marker_size": 8,
            "grid_alpha": 0.16,
            "grid_color": "#6b7280",
            "title_size": 13,
            "label_size": 12,
            "tick_size": 10,
        }
    return {
        "reported_color": "#1f77b4",
        "predicted_color": "#d62728",
        "reported_marker": "o",
        "predicted_marker": "s",
        "reported_linewidth": 2.0,
        "predicted_linewidth": 2.0,
        "marker_size": 7,
        "grid_alpha": 0.25,
        "grid_color": "#9ca3af",
        "title_size": 11,
        "label_size": 11,
        "tick_size": 10,
    }


def _apply_axes_style(ax: plt.Axes, *, pretty: bool, styles: dict[str, object]) -> None:
    if pretty:
        ax.set_facecolor("#ffffff")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#d1d5db")
        ax.spines["bottom"].set_color("#d1d5db")
        ax.tick_params(axis="both", labelsize=styles["tick_size"], colors="#374151")
        ax.grid(True, axis="y", alpha=styles["grid_alpha"], color=styles["grid_color"], linewidth=1.0)
        ax.grid(True, axis="x", alpha=0.08, color=styles["grid_color"], linewidth=0.8)
    else:
        ax.grid(True, alpha=styles["grid_alpha"])


def _plot_trace(
    ax: plt.Axes,
    trace_df: pd.DataFrame,
    *,
    participant_id: int,
    video_uid: str,
    prediction_mode: str,
    pretty: bool,
    title_participant_override: int | None,
) -> None:
    ordered = trace_df.sort_values("time_center_sec").reset_index(drop=True)
    prediction_column, prediction_label = _prediction_column_and_label(prediction_mode)
    styles = _plot_styles(pretty)
    _apply_axes_style(ax, pretty=pretty, styles=styles)
    ax.plot(
        ordered["time_center_min"],
        ordered["reported_score"],
        marker=styles["reported_marker"],
        linewidth=styles["reported_linewidth"],
        markersize=styles["marker_size"],
        markerfacecolor=styles["reported_color"],
        markeredgecolor="white" if pretty else styles["reported_color"],
        markeredgewidth=1.2 if pretty else 0.8,
        color=styles["reported_color"],
        label="Reported score",
        zorder=3,
    )
    ax.plot(
        ordered["time_center_min"],
        ordered[prediction_column],
        marker=styles["predicted_marker"],
        linewidth=styles["predicted_linewidth"],
        markersize=styles["marker_size"],
        markerfacecolor=styles["predicted_color"],
        markeredgecolor="white" if pretty else styles["predicted_color"],
        markeredgewidth=1.2 if pretty else 0.8,
        linestyle="--",
        color=styles["predicted_color"],
        label=prediction_label,
        zorder=3,
    )
    title_text = video_uid.replace("_", " ").title()
    title_participant_id = title_participant_override if title_participant_override is not None else participant_id
    ax.set_title(f"P{title_participant_id}  |  {title_text}", fontsize=styles["title_size"], pad=12)
    ax.set_xlabel("Time (min)", fontsize=styles["label_size"])
    ax.set_ylabel("Attention difficulty", fontsize=styles["label_size"])
    ax.set_ylim(0.8, 5.2)
    ax.set_yticks([1, 2, 3, 4, 5])
    if pretty:
        ax.set_xlim(
            ordered["time_center_min"].min() - 0.2,
            ordered["time_center_min"].max() + 0.2,
        )


def main() -> int:
    args = _parse_args()
    artifact_root = Path(args.artifact_root).resolve()
    run_id = str(args.run_id)
    prediction_mode = str(args.prediction_mode)
    pretty = bool(args.pretty)
    title_participant_override = args.title_participant_override
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else artifact_root / "plots" / run_id / f"engagement_traces_{prediction_mode}{'_pretty' if pretty else ''}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    selections = _resolve_selections(args.selections)
    data = _load_trace_data(artifact_root, run_id)

    n = max(1, len(selections))
    ncols = 3 if n > 4 else 2
    nrows = int(math.ceil(n / ncols))
    fig_width = 6.0 * ncols
    fig_height = 3.8 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), sharey=True)
    if pretty:
        fig.patch.set_facecolor("#ffffff")
    if hasattr(axes, "flat"):
        axes_flat = list(axes.flat)
    else:
        axes_flat = [axes]
    used_axes: list[plt.Axes] = []

    for ax, (participant_id, video_uid) in zip(axes_flat, selections, strict=False):
        trace_df = data[
            (data["participant_id"] == int(participant_id))
            & (data["video_uid"] == str(video_uid))
        ].copy()
        if trace_df.empty:
            ax.set_visible(False)
            continue
        _plot_trace(
            ax,
            trace_df,
            participant_id=participant_id,
            video_uid=video_uid,
            prediction_mode=prediction_mode,
            pretty=pretty,
            title_participant_override=title_participant_override,
        )
        used_axes.append(ax)

        single_fig, single_ax = plt.subplots(figsize=(7.2, 4.6) if pretty else (6.5, 4.0))
        if pretty:
            single_fig.patch.set_facecolor("#ffffff")
        _plot_trace(
            single_ax,
            trace_df,
            participant_id=participant_id,
            video_uid=video_uid,
            prediction_mode=prediction_mode,
            pretty=pretty,
            title_participant_override=title_participant_override,
        )
        if pretty:
            _, prediction_label = _prediction_column_and_label(prediction_mode)
            legend_handles = [
                Line2D([0], [0], color=_plot_styles(True)["reported_color"], marker=_plot_styles(True)["reported_marker"], linewidth=_plot_styles(True)["reported_linewidth"], markersize=_plot_styles(True)["marker_size"], markeredgecolor="white", markeredgewidth=1.2, label="Reported score"),
                Line2D([0], [0], color=_plot_styles(True)["predicted_color"], marker=_plot_styles(True)["predicted_marker"], linewidth=_plot_styles(True)["predicted_linewidth"], linestyle="--", markersize=_plot_styles(True)["marker_size"], markeredgecolor="white", markeredgewidth=1.2, label=prediction_label),
            ]
            single_ax.legend(
                handles=legend_handles,
                loc="upper left",
                bbox_to_anchor=(0.0, 1.02),
                ncol=2,
                frameon=False,
                fontsize=10,
            )
        else:
            single_ax.legend(loc="upper right")
        single_fig.tight_layout()
        single_path = output_dir / f"participant_{participant_id}_{video_uid}.png"
        single_fig.savefig(single_path, dpi=200, bbox_inches="tight")
        plt.close(single_fig)

    for ax in axes_flat[len(selections) :]:
        ax.set_visible(False)

    if used_axes:
        handles, labels = used_axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            fontsize=11 if pretty else 10,
        )
    mode_title = "Rounded Predictions" if prediction_mode == "rounded" else "Continuous Predictions"
    fig.suptitle(f"Reported vs Predicted Scores Over Time ({mode_title})", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    combined_path = output_dir / "selected_engagement_traces.png"
    fig.savefig(combined_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    selection_rows = pd.DataFrame(
        [{"participant_id": pid, "video_uid": vid} for pid, vid in selections]
    )
    selection_rows.to_csv(output_dir / "selected_traces.csv", index=False)
    print(f"Saved plots to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

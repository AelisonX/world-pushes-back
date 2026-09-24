import random

import matplotlib.pyplot as plt
import numpy as np

from sensing_threshold import (
    MOTION_NOISE_LEVELS,
    PRE_SLIP_LIMITS,
    collect_samples,
    evaluate_accuracy,
)


def build_accuracy_matrix():
    matrix = np.zeros(
        (
            len(PRE_SLIP_LIMITS),
            len(MOTION_NOISE_LEVELS),
        )
    )

    for i, pre_slip_limit in enumerate(
        PRE_SLIP_LIMITS
    ):
        for j, noise_level in enumerate(
            MOTION_NOISE_LEVELS
        ):
            random.seed(42)

            samples = collect_samples(
                pre_slip_displacement_limit=(
                    pre_slip_limit
                ),
                motion_noise_std=noise_level,
            )

            result = evaluate_accuracy(
                samples
            )

            if result is None:
                matrix[i, j] = np.nan
            else:
                mean_accuracy, _ = result

                matrix[i, j] = (
                    mean_accuracy
                )

    return matrix


def plot_heatmap(
    accuracy_matrix,
):
    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    image = ax.imshow(
        accuracy_matrix,
        aspect="auto",
        origin="lower",
        vmin=0.5,
        vmax=1.0,
    )

    ax.set_xticks(
        range(
            len(
                MOTION_NOISE_LEVELS
            )
        )
    )

    ax.set_xticklabels(
        [
            f"{value * 1000:.2f}"
            for value
            in MOTION_NOISE_LEVELS
        ]
    )

    ax.set_yticks(
        range(
            len(
                PRE_SLIP_LIMITS
            )
        )
    )

    ax.set_yticklabels(
        [
            f"{value * 1000:.2f}"
            for value
            in PRE_SLIP_LIMITS
        ]
    )

    ax.set_xlabel(
        "Motion sensor noise (mm)"
    )

    ax.set_ylabel(
        "Pre-slip displacement limit (mm)"
    )

    ax.set_title(
        "Pre-Transition Identifiability"
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        "Classification accuracy"
    )

    for i in range(
        len(PRE_SLIP_LIMITS)
    ):
        for j in range(
            len(MOTION_NOISE_LEVELS)
        ):
            value = accuracy_matrix[
                i,
                j,
            ]

            if np.isnan(value):
                text = "N/A"
            else:
                text = f"{value:.2f}"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
            )

    fig.tight_layout()

    output_path = (
        "sensing_threshold_heatmap.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
    )

    print(
        f"Saved: {output_path}"
    )

    plt.show()


def main():
    random.seed(42)

    accuracy_matrix = (
        build_accuracy_matrix()
    )

    print(
        "=== Accuracy matrix ==="
    )

    print(
        accuracy_matrix
    )

    plot_heatmap(
        accuracy_matrix
    )


if __name__ == "__main__":
    main()
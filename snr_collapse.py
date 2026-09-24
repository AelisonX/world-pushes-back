import random

import matplotlib.pyplot as plt

from sensing_threshold import (
    MOTION_NOISE_LEVELS,
    PRE_SLIP_LIMITS,
    collect_samples,
    evaluate_accuracy,
)


def collect_snr_results():
    results = []

    for pre_slip_limit in PRE_SLIP_LIMITS:
        for noise_level in MOTION_NOISE_LEVELS:
            random.seed(42)

            (
                samples,
                counts,
                attempts,
            ) = collect_samples(
                pre_slip_displacement_limit=pre_slip_limit,
                motion_noise_std=noise_level,
            )

            evaluation = evaluate_accuracy(
                samples
            )

            if evaluation is None:
                continue

            mean_accuracy, std_accuracy = evaluation

            if noise_level <= 0:
                continue

            snr = (
                pre_slip_limit
                / noise_level
            )

            if snr <= 0:
                continue

            results.append(
                {
                    "pre_slip_limit": pre_slip_limit,
                    "noise_level": noise_level,
                    "snr": snr,
                    "accuracy": mean_accuracy,
                    "std": std_accuracy,
                    "yield_samples": counts.get(
                        "MATERIAL_YIELD",
                        0,
                    ),
                    "slip_samples": counts.get(
                        "SUPPORT_SLIP",
                        0,
                    ),
                    "attempts": attempts,
                }
            )

    return results


def print_results(results):
    print(
        "=== Precursor-to-Noise Collapse ==="
    )

    print(
        "SNR | precursor_mm | noise_mm | "
        "accuracy | std | attempts"
    )

    print(
        "------------------------------------------------------------"
    )

    for item in sorted(
        results,
        key=lambda x: x["snr"],
    ):
        print(
            f"{item['snr']:6.2f} | "
            f"{item['pre_slip_limit'] * 1000:12.3f} | "
            f"{item['noise_level'] * 1000:8.3f} | "
            f"{item['accuracy']:8.3f} | "
            f"{item['std']:.3f} | "
            f"{item['attempts']:8d}"
        )


def plot_results(results):
    filtered_results = [
        item
        for item in results
        if item["snr"] > 0
    ]

    if not filtered_results:
        raise RuntimeError(
            "No positive SNR values are available for plotting."
        )

    sorted_results = sorted(
        filtered_results,
        key=lambda x: x["snr"],
    )

    snr_values = [
        item["snr"]
        for item in sorted_results
    ]

    accuracies = [
        item["accuracy"]
        for item in sorted_results
    ]

    errors = [
        item["std"]
        for item in sorted_results
    ]

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.errorbar(
        snr_values,
        accuracies,
        yerr=errors,
        fmt="o",
        capsize=3,
    )

    ax.axhline(
        0.5,
        linestyle="--",
    )

    ax.set_xscale(
        "log"
    )

    ax.set_ylim(
        0.45,
        1.02,
    )

    ax.set_xlabel(
        "Precursor-to-noise ratio"
    )

    ax.set_ylabel(
        "Classification accuracy"
    )

    ax.set_title(
        "Identifiability vs Precursor-to-Noise Ratio"
    )

    fig.tight_layout()

    output_path = (
        "snr_collapse.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
    )

    print(
        f"Saved: {output_path}"
    )

    plt.close(
        fig
    )


def main():
    results = collect_snr_results()

    print_results(
        results
    )

    plot_results(
        results
    )


if __name__ == "__main__":
    main()
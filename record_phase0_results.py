import csv
import random

from compliance_ablation import (
    PROBE_FORCES,
    collect_samples,
    estimate_accuracy,
)


OUTPUT_FILE = "phase0_results.csv"


def collect_rows():
    rows = []

    for support_model in [
        "rigid",
        "compliant",
    ]:
        for probe_force in PROBE_FORCES:
            random.seed(42)

            samples, transition_rate = collect_samples(
                probe_force=probe_force,
                support_model=support_model,
            )

            accuracy = estimate_accuracy(
                samples
            )

            rows.append(
                {
                    "support_model": support_model,
                    "probe_force_N": probe_force,
                    "separability_accuracy": (
                        ""
                        if accuracy is None
                        else round(accuracy, 4)
                    ),
                    "transition_rate": round(
                        transition_rate,
                        4,
                    ),
                    "safe_sample_count": len(
                        samples
                    ),
                }
            )

    return rows


def write_csv(rows):
    fieldnames = [
        "support_model",
        "probe_force_N",
        "separability_accuracy",
        "transition_rate",
        "safe_sample_count",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def print_summary(rows):
    print(
        "=== Phase 0 Results Recorder ==="
    )

    for row in rows:
        print(
            f"{row['support_model']:9} | "
            f"{row['probe_force_N']:5.1f} N | "
            f"accuracy="
            f"{row['separability_accuracy']} | "
            f"transition_rate="
            f"{row['transition_rate']} | "
            f"safe_samples="
            f"{row['safe_sample_count']}"
        )

    print()
    print(
        f"Saved results to: {OUTPUT_FILE}"
    )


def main():
    rows = collect_rows()

    write_csv(
        rows
    )

    print_summary(
        rows
    )


if __name__ == "__main__":
    main()
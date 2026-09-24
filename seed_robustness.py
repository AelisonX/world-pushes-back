import random
import statistics

from compliance_ablation import (
    collect_samples,
    estimate_accuracy,
)


SEEDS = [
    1,
    7,
    13,
    21,
    42,
    73,
    101,
    256,
    512,
    999,
]

PROBE_FORCES = [
    5.0,
    11.0,
    17.0,
    23.0,
]

SUPPORT_MODELS = [
    "rigid",
    "compliant",
]


def run_single_seed(
    seed: int,
    support_model: str,
    probe_force: float,
):
    random.seed(seed)

    samples, transition_rate = collect_samples(
        probe_force=probe_force,
        support_model=support_model,
    )

    accuracy = estimate_accuracy(
        samples
    )

    return {
        "seed": seed,
        "accuracy": accuracy,
        "transition_rate": transition_rate,
        "safe_samples": len(samples),
    }


def summarize_values(values):
    clean_values = [
        value
        for value in values
        if value is not None
    ]

    if not clean_values:
        return None

    if len(clean_values) == 1:
        return {
            "mean": clean_values[0],
            "std": 0.0,
            "min": clean_values[0],
            "max": clean_values[0],
        }

    return {
        "mean": statistics.mean(
            clean_values
        ),
        "std": statistics.stdev(
            clean_values
        ),
        "min": min(
            clean_values
        ),
        "max": max(
            clean_values
        ),
    }


def run_condition(
    support_model: str,
    probe_force: float,
):
    results = []

    for seed in SEEDS:
        result = run_single_seed(
            seed=seed,
            support_model=support_model,
            probe_force=probe_force,
        )

        results.append(
            result
        )

    accuracy_summary = summarize_values(
        [
            result["accuracy"]
            for result in results
        ]
    )

    transition_summary = summarize_values(
        [
            result["transition_rate"]
            for result in results
        ]
    )

    safe_sample_summary = summarize_values(
        [
            result["safe_samples"]
            for result in results
        ]
    )

    return {
        "support_model": support_model,
        "probe_force": probe_force,
        "results": results,
        "accuracy_summary": accuracy_summary,
        "transition_summary": transition_summary,
        "safe_sample_summary": safe_sample_summary,
    }


def print_condition_summary(
    condition,
):
    print()
    print(
        f"=== "
        f"{condition['support_model'].upper()} "
        f"| {condition['probe_force']:.1f} N "
        f"==="
    )

    print(
        "seed | accuracy | transition_rate | safe_samples"
    )

    print(
        "------------------------------------------------"
    )

    for result in condition["results"]:
        accuracy = result["accuracy"]

        if accuracy is None:
            accuracy_text = "N/A"
        else:
            accuracy_text = (
                f"{accuracy:.3f}"
            )

        print(
            f"{result['seed']:4d} | "
            f"{accuracy_text:8} | "
            f"{result['transition_rate']:.3f} | "
            f"{result['safe_samples']:12d}"
        )

    print()

    accuracy_summary = (
        condition["accuracy_summary"]
    )

    transition_summary = (
        condition["transition_summary"]
    )

    safe_sample_summary = (
        condition["safe_sample_summary"]
    )

    if accuracy_summary is not None:
        print(
            "Accuracy:"
        )

        print(
            f"  mean = "
            f"{accuracy_summary['mean']:.3f}"
        )

        print(
            f"  std  = "
            f"{accuracy_summary['std']:.3f}"
        )

        print(
            f"  min  = "
            f"{accuracy_summary['min']:.3f}"
        )

        print(
            f"  max  = "
            f"{accuracy_summary['max']:.3f}"
        )

    if transition_summary is not None:
        print(
            "Transition rate:"
        )

        print(
            f"  mean = "
            f"{transition_summary['mean']:.3f}"
        )

        print(
            f"  std  = "
            f"{transition_summary['std']:.3f}"
        )

    if safe_sample_summary is not None:
        print(
            "Safe samples:"
        )

        print(
            f"  mean = "
            f"{safe_sample_summary['mean']:.1f}"
        )


def main():
    print(
        "=== Seed Robustness Check ==="
    )

    for support_model in SUPPORT_MODELS:
        for probe_force in PROBE_FORCES:
            condition = run_condition(
                support_model=support_model,
                probe_force=probe_force,
            )

            print_condition_summary(
                condition
            )


if __name__ == "__main__":
    main()
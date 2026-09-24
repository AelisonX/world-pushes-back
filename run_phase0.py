import subprocess
import sys
from pathlib import Path


STEPS = [
    {
        "name": "Validate experiment manifest",
        "command": [
            sys.executable,
            "validate_manifest.py",
        ],
    },
    {
        "name": "Run physics regression tests",
        "command": [
            sys.executable,
            "-m",
            "pytest",
            "-q",
        ],
    },
    {
        "name": "Basic pre-transition identifiability",
        "command": [
            sys.executable,
            "identifiability.py",
        ],
    },
    {
        "name": "Information vs safety force sweep",
        "command": [
            sys.executable,
            "force_sweep.py",
        ],
    },
    {
        "name": "Rigid vs compliant support ablation",
        "command": [
            sys.executable,
            "compliance_ablation.py",
        ],
    },
    {
        "name": "Sensor ablation",
        "command": [
            sys.executable,
            "sensor_ablation.py",
        ],
    },
    {
        "name": "Minimum sensing requirement sweep",
        "command": [
            sys.executable,
            "sensing_threshold.py",
        ],
    },
    {
        "name": "Generate sensing heatmap",
        "command": [
            sys.executable,
            "plot_sensing_heatmap.py",
        ],
    },
    {
        "name": "Precursor-to-noise scaling",
        "command": [
            sys.executable,
            "snr_collapse.py",
        ],
    },
    {
        "name": "Seed robustness",
        "command": [
            sys.executable,
            "seed_robustness.py",
        ],
    },
    {
        "name": "Parameter sensitivity",
        "command": [
            sys.executable,
            "parameter_sensitivity.py",
        ],
    },
    {
        "name": "Record Phase 0 results",
        "command": [
            sys.executable,
            "record_phase0_results.py",
        ],
    },
]


def check_required_files():
    missing = []

    for step in STEPS:
        command = step["command"]

        if len(command) < 2:
            continue

        script = command[-1]

        if script.endswith(".py"):
            if not Path(script).exists():
                missing.append(
                    script
                )

    return sorted(
        set(missing)
    )


def run_step(
    index,
    total,
    name,
    command,
):
    print()
    print(
        "=" * 70
    )

    print(
        f"[{index}/{total}] {name}"
    )

    print(
        "=" * 70
    )

    print(
        "Command:",
        " ".join(command),
    )

    print()

    result = subprocess.run(
        command,
        check=False,
    )

    if result.returncode != 0:
        print()
        print(
            f"FAILED: {name}"
        )

        print(
            f"Exit code: "
            f"{result.returncode}"
        )

        return False

    print()
    print(
        f"PASSED: {name}"
    )

    return True


def main():
    print(
        "=== world-pushes-back "
        "Phase 0 Runner ==="
    )

    missing_files = (
        check_required_files()
    )

    if missing_files:
        print()
        print(
            "Cannot start."
        )

        print(
            "Missing required files:"
        )

        for file_name in missing_files:
            print(
                f"- {file_name}"
            )

        raise SystemExit(
            1
        )

    total = len(
        STEPS
    )

    passed = 0

    failed_steps = []

    for index, step in enumerate(
        STEPS,
        start=1,
    ):
        success = run_step(
            index=index,
            total=total,
            name=step["name"],
            command=step["command"],
        )

        if success:
            passed += 1
        else:
            failed_steps.append(
                step["name"]
            )

            print()
            print(
                "Stopping Phase 0 run "
                "after failure."
            )

            break

    print()
    print(
        "=" * 70
    )

    print(
        "Phase 0 summary"
    )

    print(
        "=" * 70
    )

    print(
        f"Passed: {passed}/{total}"
    )

    if failed_steps:
        print(
            "Failed steps:"
        )

        for name in failed_steps:
            print(
                f"- {name}"
            )

        raise SystemExit(
            1
        )

    print(
        "All Phase 0 steps completed."
    )


if __name__ == "__main__":
    main()
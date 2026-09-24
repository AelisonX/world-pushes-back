import json
from pathlib import Path


MANIFEST_FILE = "experiments.json"


def load_manifest():
    manifest_path = Path(
        MANIFEST_FILE
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_FILE}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_project_name(
    manifest,
):
    project_name = manifest.get(
        "project"
    )

    if not project_name:
        return [
            "Missing project name."
        ]

    return []


def validate_experiments(
    manifest,
):
    errors = []

    experiments = manifest.get(
        "experiments",
        []
    )

    if not experiments:
        errors.append(
            "No experiments found."
        )

        return errors

    seen_ids = set()

    for experiment in experiments:
        experiment_id = experiment.get(
            "id"
        )

        script = experiment.get(
            "script"
        )

        name = experiment.get(
            "name"
        )

        if not experiment_id:
            errors.append(
                f"Experiment missing id: {experiment}"
            )
            continue

        if experiment_id in seen_ids:
            errors.append(
                f"Duplicate experiment id: "
                f"{experiment_id}"
            )

        seen_ids.add(
            experiment_id
        )

        if not name:
            errors.append(
                f"{experiment_id}: missing name"
            )

        if not script:
            errors.append(
                f"{experiment_id}: missing script"
            )
            continue

        script_path = Path(
            script
        )

        if not script_path.exists():
            errors.append(
                f"{experiment_id}: "
                f"script not found: {script}"
            )

    return errors


def validate_kill_criteria(
    manifest,
):
    errors = []

    criteria = manifest.get(
        "kill_criteria",
        []
    )

    if not criteria:
        errors.append(
            "No kill criteria found."
        )

        return errors

    seen_ids = set()

    for criterion in criteria:
        criterion_id = criterion.get(
            "id"
        )

        rule = criterion.get(
            "rule"
        )

        if not criterion_id:
            errors.append(
                "Kill criterion missing id."
            )
            continue

        if criterion_id in seen_ids:
            errors.append(
                f"Duplicate kill criterion id: "
                f"{criterion_id}"
            )

        seen_ids.add(
            criterion_id
        )

        if not rule:
            errors.append(
                f"Kill criterion "
                f"{criterion_id} missing rule."
            )

    return errors


def validate_manifest():
    manifest = load_manifest()

    errors = []

    errors.extend(
        validate_project_name(
            manifest
        )
    )

    errors.extend(
        validate_experiments(
            manifest
        )
    )

    errors.extend(
        validate_kill_criteria(
            manifest
        )
    )

    return errors


def main():
    print(
        "=== Experiment Manifest Validation ==="
    )

    errors = validate_manifest()

    if errors:
        print()
        print(
            "Validation failed:"
        )

        for error in errors:
            print(
                f"- {error}"
            )

        raise SystemExit(
            1
        )

    print()
    print(
        "Manifest is valid."
    )

    print(
        "All listed experiment scripts exist."
    )


if __name__ == "__main__":
    main()
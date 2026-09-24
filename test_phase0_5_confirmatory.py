from phase0_5_dataset import (
    ObservationRecord,
)
from run_phase0_5_confirmatory import (
    apply_global_label_shuffle,
    load_manifest,
    manifest_sha256,
    validate_manifest,
)


def make_record(
    pair_id: int,
    side: str,
    label: str,
) -> ObservationRecord:
    return ObservationRecord(
        pair_id=pair_id,
        side=side,
        label=label,
        fx=1.0,
        fy=2.0,
        tip_dx=0.001,
        tip_dy=-0.001,
        rho=0.2,
        margin=0.3,
    )


def test_frozen_manifest_loads_and_validates():
    manifest = (
        load_manifest()
    )

    validate_manifest(
        manifest
    )

    assert (
        manifest[
            "manifest_version"
        ]
        == "1.1"
    )

    assert (
        manifest[
            "status"
        ]
        == "FROZEN_BEFORE_CONFIRMATORY_RUN"
    )


def test_manifest_hash_is_stable_across_calls():
    hash_a = (
        manifest_sha256()
    )

    hash_b = (
        manifest_sha256()
    )

    assert (
        hash_a
        == hash_b
    )

    assert (
        len(
            hash_a
        )
        == 64
    )


def test_global_label_shuffle_preserves_pair_balance():
    records = [
        make_record(
            pair_id=0,
            side="A",
            label="MATERIAL_YIELD",
        ),
        make_record(
            pair_id=0,
            side="B",
            label="SUPPORT_SLIP",
        ),
        make_record(
            pair_id=1,
            side="A",
            label="SUPPORT_SLIP",
        ),
        make_record(
            pair_id=1,
            side="B",
            label="MATERIAL_YIELD",
        ),
    ]

    shuffled = (
        apply_global_label_shuffle(
            records=records,
            seed=123,
        )
    )

    for pair_id in [
        0,
        1,
    ]:
        labels = {
            record.label
            for record in shuffled
            if (
                record.pair_id
                == pair_id
            )
        }

        assert labels == {
            "MATERIAL_YIELD",
            "SUPPORT_SLIP",
        }


def test_global_label_shuffle_preserves_observations():
    records = [
        make_record(
            pair_id=0,
            side="A",
            label="MATERIAL_YIELD",
        ),
        make_record(
            pair_id=0,
            side="B",
            label="SUPPORT_SLIP",
        ),
    ]

    shuffled = (
        apply_global_label_shuffle(
            records=records,
            seed=123,
        )
    )

    for (
        original,
        changed,
    ) in zip(
        records,
        shuffled,
    ):
        assert (
            original.pair_id
            == changed.pair_id
        )

        assert (
            original.side
            == changed.side
        )

        assert (
            original.fx
            == changed.fx
        )

        assert (
            original.fy
            == changed.fy
        )

        assert (
            original.tip_dx
            == changed.tip_dx
        )

        assert (
            original.tip_dy
            == changed.tip_dy
        )

        assert (
            original.rho
            == changed.rho
        )

        assert (
            original.margin
            == changed.margin
        )


def test_label_shuffle_is_reproducible():
    records = []

    for pair_id in range(
        20
    ):
        records.extend(
            [
                make_record(
                    pair_id=pair_id,
                    side="A",
                    label="MATERIAL_YIELD",
                ),
                make_record(
                    pair_id=pair_id,
                    side="B",
                    label="SUPPORT_SLIP",
                ),
            ]
        )

    shuffled_a = (
        apply_global_label_shuffle(
            records=records,
            seed=271828,
        )
    )

    shuffled_b = (
        apply_global_label_shuffle(
            records=records,
            seed=271828,
        )
    )

    assert [
        record.label
        for record in shuffled_a
    ] == [
        record.label
        for record in shuffled_b
    ]
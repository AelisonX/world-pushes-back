import math

import numpy as np

from phase0_5_bayes import (
    BayesParticle,
    bayes_support_slip_probability,
    gaussian_log_likelihood,
    logsumexp,
    sample_bayes_particles,
)
from phase0_5_dataset import (
    ObservationRecord,
)
from physics import (
    NextTransition,
)


def test_gaussian_log_likelihood_prefers_matching_mean():
    matching = (
        gaussian_log_likelihood(
            observed=1.0,
            mean=1.0,
            std=0.2,
        )
    )

    distant = (
        gaussian_log_likelihood(
            observed=1.0,
            mean=3.0,
            std=0.2,
        )
    )

    assert (
        matching
        > distant
    )


def test_gaussian_log_likelihood_rejects_nonpositive_std():
    try:
        gaussian_log_likelihood(
            observed=1.0,
            mean=1.0,
            std=0.0,
        )
    except ValueError as error:
        assert (
            "std"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected nonpositive std to fail"
        )


def test_logsumexp_matches_direct_calculation():
    values = np.array(
        [
            -2.0,
            -1.0,
            -3.0,
        ]
    )

    expected = math.log(
        sum(
            math.exp(
                value
            )
            for value in values
        )
    )

    actual = (
        logsumexp(
            values
        )
    )

    assert (
        abs(
            actual
            - expected
        )
        < 1e-12
    )


def test_particle_sampler_is_class_balanced():
    particles = (
        sample_bayes_particles(
            n_particles=20,
            seed=42,
        )
    )

    labels = [
        particle.label
        for particle in particles
    ]

    assert (
        labels.count(
            NextTransition.MATERIAL_YIELD.value
        )
        == 10
    )

    assert (
        labels.count(
            NextTransition.SUPPORT_SLIP.value
        )
        == 10
    )


def test_particle_sampler_rejects_odd_count():
    try:
        sample_bayes_particles(
            n_particles=11,
            seed=42,
        )
    except ValueError as error:
        assert (
            "even"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected odd particle count to fail"
        )


def test_bayes_probability_stays_inside_unit_interval():
    record = ObservationRecord(
        pair_id=1,
        side="A",
        label=(
            NextTransition.SUPPORT_SLIP.value
        ),
        fx=3.0,
        fy=3.0,
        tip_dx=0.001,
        tip_dy=-0.001,
        rho=0.3,
        margin=0.2,
    )

    particles = [
        BayesParticle(
            label=(
                NextTransition.MATERIAL_YIELD.value
            ),
            mean_fx=3.0,
            mean_fy=3.0,
            mean_tip_dx=0.0005,
            mean_tip_dy=-0.001,
        ),
        BayesParticle(
            label=(
                NextTransition.SUPPORT_SLIP.value
            ),
            mean_fx=3.0,
            mean_fy=3.0,
            mean_tip_dx=0.001,
            mean_tip_dy=-0.001,
        ),
    ]

    probability = (
        bayes_support_slip_probability(
            record=record,
            particles=particles,
            force_noise_std=0.2,
            motion_noise_std=0.0002,
        )
    )

    assert (
        0.0
        <= probability
        <= 1.0
    )


def test_bayes_prefers_closer_support_slip_particle():
    record = ObservationRecord(
        pair_id=1,
        side="A",
        label=(
            NextTransition.SUPPORT_SLIP.value
        ),
        fx=3.0,
        fy=3.0,
        tip_dx=0.003,
        tip_dy=-0.001,
        rho=0.3,
        margin=0.2,
    )

    particles = [
        BayesParticle(
            label=(
                NextTransition.MATERIAL_YIELD.value
            ),
            mean_fx=3.0,
            mean_fy=3.0,
            mean_tip_dx=0.0002,
            mean_tip_dy=-0.001,
        ),
        BayesParticle(
            label=(
                NextTransition.SUPPORT_SLIP.value
            ),
            mean_fx=3.0,
            mean_fy=3.0,
            mean_tip_dx=0.003,
            mean_tip_dy=-0.001,
        ),
    ]

    probability = (
        bayes_support_slip_probability(
            record=record,
            particles=particles,
            force_noise_std=0.2,
            motion_noise_std=0.0002,
        )
    )

    assert (
        probability
        > 0.5
    )


def test_identical_class_evidence_gives_half_probability():
    record = ObservationRecord(
        pair_id=1,
        side="A",
        label=(
            NextTransition.SUPPORT_SLIP.value
        ),
        fx=3.0,
        fy=3.0,
        tip_dx=0.001,
        tip_dy=-0.001,
        rho=0.3,
        margin=0.2,
    )

    particles = [
        BayesParticle(
            label=(
                NextTransition.MATERIAL_YIELD.value
            ),
            mean_fx=3.0,
            mean_fy=3.0,
            mean_tip_dx=0.001,
            mean_tip_dy=-0.001,
        ),
        BayesParticle(
            label=(
                NextTransition.SUPPORT_SLIP.value
            ),
            mean_fx=3.0,
            mean_fy=3.0,
            mean_tip_dx=0.001,
            mean_tip_dy=-0.001,
        ),
    ]

    probability = (
        bayes_support_slip_probability(
            record=record,
            particles=particles,
            force_noise_std=0.2,
            motion_noise_std=0.0002,
        )
    )

    assert (
        abs(
            probability
            - 0.5
        )
        < 1e-12
    )
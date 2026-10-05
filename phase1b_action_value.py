"""Frozen Phase 1B: same-world universal-cohort action comparison, local only."""
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from phase1a_feasibility import (
    action_specs, evaluate_pair_action, generate_target_valid_pairs,
    parameter_shift_summary, target_margin_for_world,
    gaussian_log_likelihoods, proposal_probabilities,
)
from phase0_5b_sampler_audit import sample_targeted
from physics import NextTransition, predict_next_transition, run_safe_probe

MANIFEST = Path('PHASE1B_MANIFEST.json')
PLAN = Path('PHASE1B_PLAN.md')
EXPECTED_MANIFEST_HASH = '65f8e380e092a1720923ca11389392a4b1499ca519fc7237278d496add87dad8'
EXPECTED_PLAN_HASH = '965f81f427c0e161322861501c22ba18b1b29c91f681e85a4fe012cb2ea4adf7'
RESULTS = Path('phase1b_results.json')


class GateFailure(RuntimeError):
    """A frozen acceptance gate failed; results must not be rescued."""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_digest(path, profile):
    """Reconstruct only the documented frozen text newline representation."""
    data = Path(path).read_bytes().replace(b'\r\n', b'\n')
    if profile == 'CRLF':
        data = data.replace(b'\n', b'\r\n')
    elif profile == 'LF_WITH_FINAL_CRLF':
        if data.endswith(b'\n'):
            data = data[:-1] + b'\r\n'
    else:
        raise ValueError('Unknown frozen newline profile: ' + profile)
    return hashlib.sha256(data).hexdigest()


def load_manifest():
    if (frozen_digest(MANIFEST, 'CRLF') != EXPECTED_MANIFEST_HASH
            or frozen_digest(PLAN, 'LF_WITH_FINAL_CRLF') != EXPECTED_PLAN_HASH):
        raise GateFailure('Frozen design hash changed')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    parent = json.loads(Path('PHASE1A_FEASIBILITY_MANIFEST.json').read_text())
    for key in ('action_grid', 'target', 'safety'):
        if manifest[key] != parent[key]:
            raise GateFailure('Phase 1A design changed: ' + key)
    return manifest


def require(condition, reason, diagnostics=None):
    if not condition:
        error = GateFailure(reason)
        error.diagnostics = diagnostics
        raise error


def generate_cohort(m, seed):
    """Fixed target first, then complete-pair universal legality; no noise involved."""
    specs = action_specs(m)
    target = m['target']
    pairs = generate_target_valid_pairs(m['population']['target_valid_pairs'], seed,
                                       target['reference_angle_deg'], target['epsilon_tie'], 10000)
    checks = [[evaluate_pair_action(p, a, m['safety']['rho_safe_max'], m['rejection_precedence'])
               for a in specs] for p in pairs]
    legal = np.array([[c.legal for c in row] for row in checks])
    keep = legal.all(axis=1)
    admitted = [p for p, ok in zip(pairs, keep) if ok]
    retained_checks = [c for c, ok in zip(checks, keep) if ok]
    per_action = {}
    for j, spec in enumerate(specs):
        original = [row[j] for row in checks]
        universal_worlds = [w for row in retained_checks for w in (row[j].world_a, row[j].world_b)]
        per_action[spec.action_id] = {
            'legal_pair_fraction_original': float(legal[:, j].mean()),
            'leave_one_out_gain': float(np.delete(legal, j, axis=1).all(axis=1).mean() - keep.mean()),
            'transition_rejected_fraction': float(np.mean(['TRANSITION_BEFORE_OR_AT_PROBE' in c.reasons for c in original])),
            'all_reasons_pair_counts': {reason: sum(reason in c.reasons for c in original) for reason in m['rejection_precedence']},
            'rho_min_median_p95_max': [float(x) for x in np.quantile([w.rho for w in universal_worlds], [0, .5, .95, 1])],
            'transition_margin_min_p05_median': [float(x) for x in np.quantile([w.transition_margin for w in universal_worlds], [0, .05, .5])],
        }
    shifts = parameter_shift_summary(pairs, admitted)
    before = [target_margin_for_world(w, target['reference_angle_deg']) for p in pairs for w in (p.world_a, p.world_b)]
    after = [target_margin_for_world(w, target['reference_angle_deg']) for p in admitted for w in (p.world_a, p.world_b)]
    margin_shift = abs(np.median(after) / np.median(before) - 1)
    audit = {'target_valid_pairs': len(pairs), 'universal_pairs': len(admitted), 'retention': float(keep.mean()),
             'per_action': per_action, 'parameter_shifts': shifts, 'target_margin_median_relative_shift': float(margin_shift)}
    gates = m['cohort_gates']
    require(keep.mean() >= gates['universal_legal_pair_fraction_below'], 'Universal retention')
    require(all(v['leave_one_out_gain'] < gates['single_action_leave_one_out_gain_at_or_above'] for v in per_action.values()), 'Action bottleneck')
    require(all(v['transition_rejected_fraction'] < gates['transition_before_probe_pair_fraction_at_or_above'] for v in per_action.values()), 'Transition rejection')
    require(all(abs(v['standardized_mean_difference']) < gates['absolute_standardized_mean_difference_at_or_above'] for v in shifts.values()), 'Hidden-parameter shift')
    require(margin_shift < gates['target_margin_median_relative_shift_at_or_above'], 'Target-margin shift')
    interior = np.array([all(w.rho < m['controls']['interior_rho_max'] and w.transition_margin >= m['controls']['interior_margin_min']
                            for c in row for w in (c.world_a, c.world_b)) for row in retained_checks])
    return admitted, interior, audit


def split_pairs(pairs, fraction, seed):
    order = np.random.default_rng(seed).permutation(len(pairs))
    n = int(len(pairs) * fraction)
    require(0 < n < len(pairs), 'Empty split')
    train, test = order[:n], order[n:]
    train_ids = {pairs[i].pair_id for i in train}
    test_ids = {pairs[i].pair_id for i in test}
    require(not train_ids & test_ids, 'Pair leakage')
    def signatures(indices):
        return {tuple(asdict(w).values()) for i in indices for w in (pairs[i].world_a, pairs[i].world_b)}
    require(not signatures(train) & signatures(test), 'World parameter signature leakage')
    return train, test


def legal_means(pairs, specs, support_model='compliant'):
    """Shape pair x twin x action x 4; only legal observations leave the simulator."""
    means = np.empty((len(pairs), 2, len(specs), 4))
    labels = np.empty((len(pairs), 2), dtype=int)
    for i, pair in enumerate(pairs):
        for side, (world, label) in enumerate(((pair.world_a, pair.label_a), (pair.world_b, pair.label_b))):
            require(predict_next_transition(world, 45.0)[0] == label, 'Target changed')
            labels[i, side] = int(label == NextTransition.SUPPORT_SLIP)
            for j, spec in enumerate(specs):
                result = run_safe_probe(world, spec.action(), force_noise_std=0, motion_noise_std=0, support_model=support_model)
                require(result.still_blocked, 'Probe transitioned')
                o = result.observation
                means[i, side, j] = (o.fx, o.fy, o.tip_dx, o.tip_dy)
    require(np.all(labels.sum(axis=1) == 1), 'Class balance broken')
    return means, labels


def observation_noise(n_pairs, n_actions, seed, shared=True):
    shape = (n_pairs, 1, 1 if shared else n_actions, 4)
    return np.random.default_rng(seed).standard_normal(shape)


def class_bank(means, labels, action_index):
    rows = means[:, :, action_index, :].reshape(-1, 4)
    y = labels.reshape(-1)
    return [rows[y == c] for c in (0, 1)]


def log_posterior(observations, banks, std):
    """Exact full empirical-bank Gaussian evidence; no importance resampling."""
    require(all(len(b) > 0 for b in banks), 'Empty class bank')
    evidence = np.empty((len(observations), 2))
    for start in range(0, len(observations), 64):
        obs = observations[start:start + 64]
        for c, bank in enumerate(banks):
            # All four official channels retained. Common normalisation cancels.
            residual = (obs[:, None, :] - bank[None, :, :]) / std
            logs = -.5 * np.sum(residual * residual, axis=2)
            maximum = logs.max(axis=1)
            evidence[start:start + len(obs), c] = maximum + np.log(np.exp(logs - maximum[:, None]).sum(axis=1)) - np.log(len(bank))
    return evidence - np.logaddexp(evidence[:, 0], evidence[:, 1])[:, None]


def information_scores(labels, log_probs):
    return 1 + log_probs[np.arange(len(labels)), labels] / np.log(2)


def evaluate(means_train, y_train, means_test, y_test, noise, std):
    logs = []
    for j in range(means_train.shape[2]):
        observations = (means_test[:, :, j, :] + noise[:, :, 0 if noise.shape[2] == 1 else j, :] * std).reshape(-1, 4)
        logs.append(log_posterior(observations, class_bank(means_train, y_train, j), std))
    return np.stack(logs, axis=1)  # world x action x class


def pair_scores(y, logs):
    return np.stack([information_scores(y.reshape(-1), logs[:, j]) for j in range(logs.shape[1])], axis=1).reshape(len(y), 2, -1).mean(axis=1)


def secondary(y, logs):
    labels = y.reshape(-1)
    result = []
    for j in range(logs.shape[1]):
        lp = logs[:, j]
        probabilities = np.exp(lp)
        result.append({'roc_auc': float(roc_auc_score(labels, probabilities[:, 1])),
                       'log_loss_nats': float(-lp[np.arange(len(labels)), labels].mean()),
                       'brier': float(np.mean((probabilities[:, 1] - labels) ** 2)),
                       'posterior_entropy_bits': float(-np.mean(np.sum(probabilities * lp, axis=1)) / np.log(2))})
    return result


def contrasts(scores, draws, seed):
    combos = list(itertools.combinations(range(scores.shape[1]), 2))
    means = scores.mean(axis=0)
    differences = np.array([means[a] - means[b] for a, b in combos])
    rng = np.random.default_rng(seed)
    boot = np.stack([scores[rng.integers(len(scores), size=len(scores))].mean(axis=0) for _ in range(draws)])
    errors = np.stack([boot[:, a] - boot[:, b] for a, b in combos], axis=1) - differences
    radius = float(np.quantile(np.max(np.abs(errors), axis=1), .95))
    return [{'a': a, 'b': b, 'difference_bits': float(d), 'simultaneous_ci': [float(d-radius), float(d+radius)]}
            for (a, b), d in zip(combos, differences)], np.quantile(boot, [.025, .975], axis=0).T.tolist()


def numerical_audit(m, rep, mt, yt, me, ye, noise, full_logs):
    cfg = m['numerical']
    std = np.array(m['noise']['std'])
    rows = np.linspace(0, len(me)*2-1, min(cfg['audit_rows'], len(me)*2), dtype=int)
    full_scores = pair_scores(ye, full_logs)
    half_results = []
    for indices in (np.arange(0, len(mt), 2), np.arange(1, len(mt), 2)):
        half = evaluate(mt[indices], yt[indices], me, ye, noise, std)
        errors = np.abs(np.exp(half[:, :, 1]) - np.exp(full_logs[:, :, 1]))
        bits = np.abs(pair_scores(ye, half).mean(axis=0) - full_scores.mean(axis=0))
        summary = [{'primary_mean_difference_bits': float(bits[j]), 'posterior_p95_error': float(np.quantile(errors[:, j], .95)),
                    'posterior_max_error': float(errors[:, j].max())} for j in range(mt.shape[2])]
        half_results.append(summary)
        for s in summary:
            require(s['primary_mean_difference_bits'] <= cfg['bank_half_mean_bits_max'] and
                    s['posterior_p95_error'] <= cfg['bank_half_posterior_p95_max'] and
                    s['posterior_max_error'] <= cfg['bank_half_posterior_max'], 'Independent bank-half numerical stability', {'half_results_so_far':half_results})
    actions = []
    for j in range(mt.shape[2]):
        observations = (me[:, :, j] + noise[:, :, 0] * std).reshape(-1, 4)
        banks = class_bank(mt, yt, j)
        esses = [[], []]
        errors, score_errors = [], []
        for k, row in enumerate(rows):
            obs = observations[row]
            likelihoods = [gaussian_log_likelihoods(obs, bank, std) for bank in banks]
            qs = [proposal_probabilities(bank[:, 2], obs[2], cfg['alpha'], cfg['tau_m']) for bank in banks]
            for repeat in range(cfg['replicates']):
                sampled = [sample_targeted(likelihoods[c], qs[c], cfg['draws_per_class'],
                           int(np.random.SeedSequence([rep['sampler_seed'], j, k, repeat, c]).generate_state(1)[0])) for c in (0, 1)]
                le = np.array([s['log_evidence'] for s in sampled])
                lp = le - np.logaddexp(*le)
                errors.append(abs(np.exp(lp[1]) - np.exp(full_logs[row, j, 1])))
                label = ye.reshape(-1)[row]
                score_errors.append(abs(lp[label] - full_logs[row, j, label]) / np.log(2))
                for c in (0, 1):
                    esses[c].append(sampled[c]['ess_fraction'])
        summary = {'ess': {str(c): {'p05':float(np.quantile(esses[c], .05)), 'median':float(np.median(esses[c]))} for c in (0,1)},
                   'sampled_posterior_p95_error': float(np.quantile(errors,.95)), 'sampled_posterior_max_error':float(max(errors)),
                   'sampled_score_abs_error_mean_bits':float(np.mean(score_errors)), 'sampled_score_abs_error_max_bits':float(max(score_errors))}
        actions.append(summary)
        require(all(s['p05']>=cfg['ess_p05_min'] and s['median']>=cfg['ess_median_min'] for s in summary['ess'].values()), 'Per-action realized ESS', {'actions_so_far':actions})
        require(summary['sampled_posterior_p95_error']<=cfg['sampled_posterior_p95_max'] and summary['sampled_posterior_max_error']<=cfg['sampled_posterior_max'], 'Targeted sampler posterior stability', {'actions_so_far':actions})
    return {'bank_halves': half_results, 'realized_sampler': actions, 'pass': True}


def null_control(scores, threshold, rigid=False):
    values = scores.mean(axis=0)
    amplitude = np.max(np.abs(values)) if rigid else max(0., np.max(values))
    return {'values_bits': values.tolist(), 'span_bits':float(np.ptp(values)),
            'pass':bool(amplitude < threshold and np.ptp(values) < threshold)}


def run_rep(m, rep):
    print(rep['id'] + ': building universally legal worlds and fixed split', flush=True)
    pairs, interior, cohort = generate_cohort(m, rep['world_seed'])
    train, test = split_pairs(pairs, m['population']['train_fraction'], rep['split_seed'])
    require(interior[test].mean() >= m['controls']['interior_retention_min'], 'Interior subset too small')
    specs = action_specs(m)
    means, y = legal_means(pairs, specs)
    mt, yt, me, ye = means[train], y[train], means[test], y[test]
    noise = observation_noise(len(test), len(specs), rep['noise_seed'])
    std = np.array(m['noise']['std'])
    logs = evaluate(mt, yt, me, ye, noise, std)
    print(rep['id'] + ': checking numerical gates before revealing action values', flush=True)
    numerical = numerical_audit(m, rep, mt, yt, me, ye, noise, logs)
    rigid, ry = legal_means(pairs, specs, 'rigid')
    rigid_logs = evaluate(rigid[train], ry[train], rigid[test], ry[test], noise, std)
    rigid_control = null_control(pair_scores(ye, rigid_logs), m['controls']['null_bits_max'], rigid=True)
    require(rigid_control['pass'], 'Rigid null')
    shuffle_rng = np.random.default_rng(rep['shuffle_seed'])
    shuffled_train = np.bitwise_xor(yt, shuffle_rng.integers(2, size=(len(yt), 1)))
    shuffled_test = np.bitwise_xor(ye, shuffle_rng.integers(2, size=(len(ye), 1)))
    shuffle_logs = evaluate(mt, shuffled_train, me, shuffled_test, noise, std)
    shuffle_control = null_control(pair_scores(shuffled_test, shuffle_logs), m['controls']['null_bits_max'])
    require(shuffle_control['pass'], 'Label shuffle')
    dup = log_posterior((me[:,:,4] + noise[:,:,0]*std).reshape(-1,4), class_bank(mt,yt,4), std)
    duplicate_error = float(np.abs(dup - logs[:,4]).max())
    require(duplicate_error <= m['controls']['duplicate_tolerance'], 'Same-action duplicate')
    primary = pair_scores(ye, logs)
    comparisons, cis = contrasts(primary, m['bootstrap_draws'], rep['bootstrap_seed'])
    robust = {}
    independent_noise = observation_noise(len(test),len(specs),rep['independent_noise_seed'],shared=False)
    robust['independent_noise'] = pair_scores(ye,evaluate(mt,yt,me,ye,independent_noise,std)).mean(axis=0).tolist()
    for scale in m['noise']['sd_multipliers']:
        robust['noise_sd_' + str(scale)] = pair_scores(ye,evaluate(mt,yt,me,ye,noise,std*scale)).mean(axis=0).tolist()
    robust['interior_subset'] = primary[interior[test]].mean(axis=0).tolist()
    report = {'id':rep['id'],'cohort':cohort,'train_pairs':len(train),'test_pairs':len(test),'interior_test_retention':float(interior[test].mean()),
              'numerical':numerical,'controls':{'rigid_null':rigid_control,'label_shuffle':shuffle_control,'duplicate_max_log_posterior_difference':duplicate_error,'world_split_integrity':True},
              'actions':[dict(action_id=a.action_id,force_N=a.force_N,angle_deg=a.angle_deg,information_lower_bound_bits=float(primary[:,j].mean()),marginal_ci=cis[j],**secondary(ye,logs)[j]) for j,a in enumerate(specs)],
              'contrasts':comparisons,'robustness_primary_bits':robust}
    # Local audit only: world identities, frozen target labels and action-wide split.
    rows = [{'pair_id': rep['id'] + ':' + str(p.pair_id), 'split': 'train' if i in set(train) else 'test',
             'world_a':asdict(p.world_a), 'world_b':asdict(p.world_b), 'labels':y[i].tolist()} for i,p in enumerate(pairs)]
    Path('phase1b_' + rep['id'].lower() + '_worlds.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    np.savez_compressed('phase1b_' + rep['id'].lower() + '_records.npz', train_indices=train,test_indices=test,
                        legal_observation_means=means,labels=y,standardized_noise=noise,log_posteriors=logs,
                        primary_pair_scores=primary,interior=interior)
    return report


def decide(reports, delta):
    meaningful = []
    for i,c in enumerate(reports[0]['contrasts']):
        intervals = [r['contrasts'][i]['simultaneous_ci'] for r in reports]
        sign = 1 if all(lo > delta for lo,hi in intervals) else -1 if all(hi < -delta for lo,hi in intervals) else 0
        if sign:
            meaningful.append((c['a'],c['b'],sign))
    if meaningful:
        for a,b,sign in meaningful:
            for r in reports:
                require(all(sign*(v[a]-v[b]) > delta for v in r['robustness_primary_bits'].values()), 'Meaningful contrast failed frozen robustness')
        return 'PHASE1B_DIAGNOSTIC_ACTION_DIFFERENCE', meaningful
    negative = all(all(lo >= -delta and hi <= delta for lo,hi in [c['simultaneous_ci']]) for r in reports for c in r['contrasts'])
    return ('CURRENT_ACTION_GRID_TOO_SIMILAR_TO_DISTINGUISH_MEANINGFULLY' if negative else 'PHASE1B_INCONCLUSIVE'), []


def main():
    m = load_manifest()
    artifact = {'status':'RUNNING','public_claims':'NONE UNTIL HUMAN REVIEW','baseline_commit':m['baseline_commit'],
                'manifest_sha256':digest(MANIFEST),'plan_sha256':digest(PLAN),'source_sha256':digest(__file__),
                'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'reports':[]}
    try:
        for rep in m['repetitions']:
            artifact['reports'].append(run_rep(m,rep))
        artifact['status'],artifact['meaningful_contrasts'] = decide(artifact['reports'],m['minimum_meaningful_effect_bits'])
    except GateFailure as error:
        artifact['status'] = 'PHASE1B_STOP_GATE_FAILURE'
        artifact['reason'] = str(error)
        artifact['gate_diagnostics'] = getattr(error, 'diagnostics', None)
        # Earlier completed repetitions are quarantined, not accepted rankings.
        artifact['accepted_action_ranking'] = False
    RESULTS.write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(artifact['status'], artifact.get('reason',''), flush=True)
    if artifact['status'] == 'PHASE1B_STOP_GATE_FAILURE':
        raise SystemExit(2)


if __name__ == '__main__':
    main()

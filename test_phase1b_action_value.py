"""Phase 1B structural/regression tests; no frozen confirmatory seeds used."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from phase1b_action_value import (
    GateFailure, load_manifest, generate_cohort, split_pairs, legal_means,
    observation_noise, class_bank, log_posterior, information_scores,
    pair_scores, evaluate, contrasts, null_control, decide,
)
from phase1a_feasibility import action_specs


def fixture():
    m=copy.deepcopy(load_manifest())
    m['population']['target_valid_pairs']=30
    pairs, interior, audit=generate_cohort(m, 1177)
    return m,pairs,interior,audit


def test_frozen_action_target_matches_parent():
    m=load_manifest()
    assert len(action_specs(m)) == 9
    assert m['target']['reference_angle_deg'] == 45
    assert m['minimum_meaningful_effect_bits'] == .02


def test_universal_legality_and_fixed_class_balance():
    m,pairs,interior,audit=fixture()
    means, y = legal_means(pairs,action_specs(m))
    assert means.shape == (len(pairs),2,9,4)
    assert np.all(y.sum(axis=1) == 1)
    assert audit['retention'] >= .5
    # Changing probe angle does not use the probe's angle-specific label.
    assert len({tuple(row) for row in y}) <= 2


def test_actions_and_twins_share_one_split():
    m,pairs,_,_=fixture()
    train,test=split_pairs(pairs,.6,1178)
    assert not set(train)&set(test)
    assert set(train)|set(test) == set(range(len(pairs)))
    means,y=legal_means(pairs,action_specs(m))
    assert means[train].shape[2] == means[test].shape[2] == 9
    assert np.all(y[train].sum(axis=1)==1)
    assert np.all(y[test].sum(axis=1)==1)


def test_duplicate_hidden_worlds_cannot_cross_split():
    _,pairs,_,_=fixture()
    duplicated=[pairs[0],replace(pairs[0],pair_id=999)]
    with pytest.raises(GateFailure,match='signature leakage'):
        split_pairs(duplicated,.5,1180)


def test_noise_coupling_and_independent_action_schedule():
    shared=observation_noise(5,9,1181)
    assert shared.shape == (5,1,1,4)
    assert np.array_equal(shared,observation_noise(5,9,1181))
    independent=observation_noise(5,9,1181,shared=False)
    assert independent.shape == (5,1,9,4)
    assert not np.array_equal(independent[:,:,0],independent[:,:,1])


def test_primary_is_proper_log_score_and_not_entropy():
    y=np.array([0,1])
    assert np.allclose(information_scores(y,np.log([[.5,.5],[.5,.5]])),0)
    assert np.all(information_scores(y,np.log([[.1,.9],[.9,.1]])) < 0)
    assert np.all(information_scores(y,np.log([[.9,.1],[.1,.9]])) > 0)


def test_known_gaussian_posterior_and_force_cancellation():
    banks=[np.array([[3.,2.,0.,0.]]),np.array([[3.,2.,1.,0.]])]
    std=np.ones(4)
    obs=np.array([[3.,2.,0.,0.]])
    posterior=log_posterior(obs,banks,std)
    assert np.isclose(np.exp(posterior[0,1]),1/(1+np.exp(.5)))
    obs[0,:2]=[7.,8.]
    assert np.allclose(posterior,log_posterior(obs,banks,std))


def test_same_action_and_rigid_null():
    m,pairs,_,_=fixture()
    means,y=legal_means(pairs,action_specs(m),'rigid')
    train,test=split_pairs(pairs,.6,1182)
    noise=observation_noise(len(test),9,1183)
    std=np.array(m['noise']['std'])
    logs=evaluate(means[train],y[train],means[test],y[test],noise,std)
    assert null_control(pair_scores(y[test],logs),.02,rigid=True)['pass']
    banks=class_bank(means[train],y[train],4)
    obs=(means[test,:,4]+noise[:,:,0]*std).reshape(-1,4)
    assert np.array_equal(logs[:,4],log_posterior(obs,banks,std))


def test_pair_bootstrap_identical_actions_zero_contrast():
    scores=np.tile(np.array([[.1],[.2],[.3]]),(1,9))
    cs,_=contrasts(scores,100,1184)
    assert len(cs)==36
    assert all(c['difference_bits']==0 and c['simultaneous_ci']==[0.,0.] for c in cs)


def test_decision_negative_inconclusive_success_and_stop():
    def report(interval):
        return {'contrasts':[{'a':0,'b':1,'simultaneous_ci':interval}],
                'robustness_primary_bits':{'independent_noise':[.1,0.], 'interior_subset':[.1,0.]}}
    assert decide([report([-.01,.01]),report([-.01,.01])],.02)[0].startswith('CURRENT_ACTION_GRID')
    assert decide([report([-.03,.03]),report([-.03,.03])],.02)[0]=='PHASE1B_INCONCLUSIVE'
    assert decide([report([.03,.05]),report([.03,.05])],.02)[0]=='PHASE1B_DIAGNOSTIC_ACTION_DIFFERENCE'
    bad=report([.03,.05]); bad['robustness_primary_bits']['interior_subset']=[0.,.1]
    with pytest.raises(GateFailure,match='robustness'):
        decide([report([.03,.05]),bad],.02)


@pytest.mark.parametrize('name,profile,expected', [
    ('PHASE1B_MANIFEST.json', 'CRLF', '65f8e380e092a1720923ca11389392a4b1499ca519fc7237278d496add87dad8'),
    ('PHASE1B_PLAN.md', 'LF_WITH_FINAL_CRLF', '965f81f427c0e161322861501c22ba18b1b29c91f681e85a4fe012cb2ea4adf7'),
])
@pytest.mark.parametrize('checkout', ['LF', 'CRLF', 'historical'])
def test_frozen_digest_checkout_portability(monkeypatch, name, profile, expected, checkout):
    from pathlib import Path
    from phase1b_action_value import frozen_digest
    data = Path(name).read_bytes().replace(b'\r\n', b'\n')
    if checkout == 'CRLF':
        data = data.replace(b'\n', b'\r\n')
    elif checkout == 'historical':
        data = (data.replace(b'\n', b'\r\n') if profile == 'CRLF'
                else data[:-1] + b'\r\n')
    monkeypatch.setattr(Path, 'read_bytes', lambda path: data)
    assert frozen_digest(name, profile) == expected


@pytest.mark.parametrize('target', ['PHASE1B_MANIFEST.json', 'PHASE1B_PLAN.md'])
@pytest.mark.parametrize('mutation', ['content', 'space', 'missing_final_newline', 'bare_CR'])
def test_frozen_gate_rejects_other_changes(monkeypatch, target, mutation):
    from pathlib import Path
    original = Path.read_bytes
    def changed(path):
        data = original(path).replace(b'\r\n', b'\n')
        if path.name == target:
            if mutation == 'content':
                data = data.replace(b'Phase', b'phase', 1)
                # The manifest may use lowercase keys rather than a Phase title.
                if data == original(path).replace(b'\r\n', b'\n'):
                    data = b'X' + data[1:]
            elif mutation == 'space':
                data = b' ' + data
            elif mutation == 'missing_final_newline':
                data = data[:-1]
            else:
                data = data.replace(b'\n', b'\r', 1)
        return data
    monkeypatch.setattr(Path, 'read_bytes', changed)
    with pytest.raises(GateFailure, match='Frozen design hash changed'):
        load_manifest()


def test_frozen_gate_accepts_lf_checkout(monkeypatch):
    from pathlib import Path
    original = Path.read_bytes
    monkeypatch.setattr(Path, 'read_bytes', lambda path: original(path).replace(b'\r\n', b'\n'))
    assert load_manifest()['minimum_meaningful_effect_bits'] == .02


def test_unknown_frozen_profile_rejected():
    from phase1b_action_value import frozen_digest
    with pytest.raises(ValueError, match='Unknown frozen newline profile'):
        frozen_digest('PHASE1B_MANIFEST.json', 'unknown')

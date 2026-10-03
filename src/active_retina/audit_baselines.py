"""Reproducible development audit; never evaluates historical test patients."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .evaluate_active_retina_rc_policy import evaluate_policy, pair_predictions, read_predictions
from .write_fast_paper_pack import (
    BRANCHES, candidate_grid, confidence_triage_metrics, select_confidence_baseline,
    threshold_metrics, write_csv,
)


def main():
    seed = 20261003
    rows, comparisons, audit = [], [], []
    for branch in BRANCHES:
        source = branch['predictions']
        raw = read_predictions(source)
        assert all(r['split'] in {'train', 'calibration', 'validation'} for r in raw)
        assert all(r['prediction_scope'] in {'crossfit', 'heldout_development'} for r in raw)
        patients = {s: {r['patient_id'] for r in raw if r['split'] == s}
                    for s in ('train', 'calibration', 'validation')}
        assert not any(patients[a] & patients[b] for a, b in
                       [('train', 'calibration'), ('train', 'validation'), ('calibration', 'validation')])
        pairs = list(pair_predictions(raw, branch['first_view'], branch['second_view']).values())
        cal = [r for r in pairs if r['split'] == 'calibration']
        val = [r for r in pairs if r['split'] == 'validation']
        policy = json.loads(branch['policy'].read_text())
        audit.append({'branch': branch['name'], 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                      'patients': {s: len(p) for s, p in patients.items()},
                      'calibration_eyes': len(cal), 'validation_eyes': len(val),
                      'historical_test_evaluated': False})
        for target in (0.01, 0.05, 0.10):
            selected = next(p for p in policy['selected_policies'] if p['target_risk'] == target)
            funcs = {}
            if selected['status'] == 'EMPIRICALLY_SELECTED_UNCERTIFIED':
                funcs['acquisition'] = lambda ex, p=selected: evaluate_policy(
                    ex, p['stop_low'], p['stop_high'], p['final_threshold'])
            triage = select_confidence_baseline(cal, val, target, 0.5)
            if triage['status'] == 'EMPIRICALLY_SELECTED_UNCERTIFIED':
                funcs['confidence_triage'] = lambda ex, p=triage: confidence_triage_metrics(
                    ex, p['stop_low'], p['stop_high'])
            for name, key, images in [('fixed_one', 'first_score', 1), ('fixed_two', 'two_view_score', 2)]:
                candidates = [(t, threshold_metrics(cal, key, t, images)) for t in
                              sorted(set([0.0, 1.0] + candidate_grid([x[key] for x in cal], 10000)))]
                feasible = [(t, m) for t, m in candidates
                            if m['missed_referable_risk_decided'] <= target and m['specificity_decided'] >= 0.5]
                pool = feasible or [(t, m) for t, m in candidates if m['missed_referable_risk_decided'] <= target]
                t, _ = min(pool, key=lambda z: (-z[1]['specificity_decided'], z[0]))
                funcs[name] = lambda ex, k=key, t=t, i=images: threshold_metrics(ex, k, t, i)
            def metrics(fn, ex):
                m = fn(ex)
                return {'images': m['mean_images'], 'defer': m['defer_rate'],
                        'risk': m.get('missed_referable_dr_risk', m.get('missed_referable_risk_decided')),
                        'specificity': m.get('specificity_among_decided', m.get('specificity_decided'))}
            clusters = {p: [x for x in val if x['patient_id'] == p] for p in sorted(patients['validation'])}
            rng = np.random.default_rng(seed)
            draws = [[x for p in rng.choice(list(clusters), size=len(clusters)) for x in clusters[p]]
                     for _ in range(1000)]
            evaluated = {}
            for name, fn in funcs.items():
                cm, vm = metrics(fn, cal), metrics(fn, val)
                reps = [metrics(fn, d) for d in draws]
                evaluated[name] = (vm, reps)
                row = {'branch': branch['name'], 'calibration_target': target, 'method': name,
                       'calibration_feasible': cm['risk'] <= target and cm['specificity'] >= 0.5,
                       **vm}
                for key in vm:
                    finite = [r[key] for r in reps if math.isfinite(r[key])]
                    lo, hi = np.quantile(finite, [0.025, 0.975]) if finite else (float('nan'), float('nan'))
                    row[key + '_ci_low'], row[key + '_ci_high'] = float(lo), float(hi)
                rows.append(row)
            if 'acquisition' in evaluated and 'confidence_triage' in evaluated:
                a, ar = evaluated['acquisition']; t, tr = evaluated['confidence_triage']
                for cost in (0, 1, 2, 5, 10):
                    delta = a['images'] - t['images'] + cost * (a['defer'] - t['defer'])
                    reps = [x['images'] - y['images'] + cost * (x['defer'] - y['defer']) for x, y in zip(ar, tr)]
                    lo, hi = np.quantile(reps, [0.025, 0.975])
                    comparisons.append({'branch': branch['name'], 'calibration_target': target,
                                        'deferral_cost': cost, 'delta_cost_acquisition_minus_triage': delta,
                                        'ci_low': float(lo), 'ci_high': float(hi)})
    write_csv(Path('results/audited_baselines.csv'), rows)
    write_csv(Path('results/audited_paired_cost_comparisons.csv'), comparisons)
    Path('results/manuscript_evidence_audit.json').write_text(json.dumps({
        'seed': seed, 'bootstrap_replicates': 1000, 'sources': audit,
        'evaluation_status': 'REUSED_DEVELOPMENT_VALIDATION_NOT_CONFIRMATORY',
        'finding': 'Shared calibration targets; infeasible fixed baselines shown with risk-constrained fallback explicitly marked.',
        'uncertainty': 'Fixed-model fixed-threshold paired patient bootstrap; excludes selection and training uncertainty.'
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()

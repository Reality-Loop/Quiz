"""Deterministic noisy OBSERVATION fixtures. Not recorded videos or expert ground truth."""
from __future__ import annotations
import hashlib
import random
from pathlib import Path
from .schema import ACTIONS, STATES, TARGETS, ACTION_TO_STEP, write_jsonl, save_json


def distribution(rng, true_label, choices, noise, force=False):
    winner = true_label
    if not force and rng.random() < noise:
        winner = rng.choice([x for x in choices if x != true_label])
    confidence = rng.uniform(.69, .94) if winner == true_label else rng.uniform(.51, .84)
    remaining = [rng.uniform(.2, 1) for _ in range(len(choices) - 1)]
    denom = sum(remaining)
    result, j = {}, 0
    for x in choices:
        if x == winner:
            result[x] = confidence
        else:
            result[x] = (1 - confidence) * remaining[j] / denom
            j += 1
    return result


def protocol(mix_required=True):
    return {'protocol_id': 'mock_transfer_v1' if mix_required else 'mock_transfer_no_mix_v1',
            'target_id': 'tube_a', 'mix_required': mix_required,
            'steps': [{'id': 'S1', 'name': 'Prepare a fresh tip', 'actions': ['attach_tip']},
                      {'id': 'S2', 'name': 'Transfer to the assigned target', 'actions': ['aspirate', 'dispense']},
                      {'id': 'S3', 'name': 'Mix when required', 'actions': ['mix']},
                      {'id': 'S4', 'name': 'Finish the transfer', 'actions': ['eject_tip']}]}


def make_episode(seed: int, split: str, index: int):
    rng = random.Random(seed)
    eid = hashlib.sha256(f'episode-{seed}'.encode()).hexdigest()[:14]
    wrong = index % 4 in (1, 3)
    omitted = index % 4 in (2, 3)
    required = index % 7 != 6
    no_opportunity = wrong and index % 5 == 1
    plan = [('background', rng.randint(2, 4)), ('attach_tip', rng.randint(3, 5)),
            ('aspirate', rng.randint(5, 8)), ('dispense', rng.randint(4, 7))]
    if not omitted:
        plan.append(('mix', rng.randint(4, 6)))
    plan.extend([('eject_tip', rng.randint(3, 5)), ('background', rng.randint(2, 4))])
    actions = [action for action, length in plan for _ in range(length)]
    dstart = actions.index('dispense')
    dend = dstart + actions.count('dispense')
    estart = actions.index('eject_tip')
    # A no-opportunity case has visually reliable evidence only after irreversible mock delivery.
    td = dstart + 2 if no_opportunity else dstart - 2
    pnr = dstart + 1
    frames, observations = [], []
    current_state = 'empty'
    ood = split == 'dev_ood'
    for t, action in enumerate(actions):
        if action == 'aspirate':
            current_state = 'loaded'
        elif action == 'dispense':
            current_state = 'transferred'
        elif action == 'mix':
            current_state = 'mixed'
        frames.append({'t': float(t), 'action': action, 'step_id': ACTION_TO_STEP[action], 'state': current_state})
        target = 'tube_b' if wrong and td <= t < dend else 'tube_a'
        if wrong and no_opportunity and dstart - 2 <= t < td:
            target = 'unknown'
        views = []
        for cam in range(2):
            if rng.random() < .05:
                continue
            visible = rng.random() > (.14 if cam == 0 else .10)
            noise = (.26 if cam == 0 else .11) + (.09 if ood else 0)
            ad = distribution(rng, action, (*ACTIONS, 'unknown'), noise)
            sd = distribution(rng, current_state, (*STATES, 'unknown'), noise * .6)
            target_scores = distribution(rng, target, TARGETS, noise * .35,
                                         force=(wrong and dstart - 2 <= t < td))
            # Ensure the documented earliest reliable evidence exists in at least one camera.
            if cam == 1 and wrong and t == td:
                visible = True
                target_scores = distribution(rng, 'tube_b', TARGETS, 0, force=True)
            if not visible:
                ad, sd, target_scores = {'unknown': 1.0}, {'unknown': 1.0}, {'unknown': 1.0}
            views.append({'camera_id': f'cam_{cam+1}', 'source_t': float(t),
                          'evidence_id': f'{eid}:cam_{cam+1}:{t:04d}', 'visible': visible,
                          'action_scores': ad, 'state_scores': sd, 'target_scores': target_scores})
        observations.append({'t': float(t), 'views': views})
    events = []
    if wrong:
        # null PNR is deliberately present; no invented irreversible timestamp.
        effective_pnr = None if index % 9 == 5 else float(pnr)
        events.append({'type': 'wrong_target', 'scope_id': 'tube_b', 't_occur': float(dstart),
                       't_detectable': float(td), 't_pnr': effective_pnr, 't_end': float(dend),
                       'severity': 'high',
                       'evidence_ids': [v['evidence_id'] for v in observations[td]['views'] if v['visible']],
                       'pnr_rationale': 'Synthetic timing fixture only; not a scientific claim.'})
    if omitted and required:
        events.append({'type': 'missing_mix', 'scope_id': 'tube_a', 't_occur': float(estart),
                       't_detectable': float(estart), 't_pnr': None, 't_end': float(len(actions)),
                       'severity': 'medium',
                       'evidence_ids': [v['evidence_id'] for v in observations[estart]['views'] if v['visible']],
                       'pnr_rationale': None})
    ep = {'schema_version': '1.0', 'dataset_kind': 'synthetic_fixture', 'episode_id': eid,
          'trial_id': f'trial_{eid}', 'operator_id': f'operator_{"C" if ood else "AB"}_{index%4}',
          'lab_id': 'mock_lab_C' if ood else ('mock_lab_A' if index % 2 else 'mock_lab_B'),
          'duration_s': float(len(actions)), 'protocol': protocol(required), 'observations': observations}
    gt = {'episode_id': eid, 'annotation_source': 'synthetic_program', 'frames': frames, 'events': events}
    return ep, gt


def generate(root: Path, seed=20260914, train=24, dev=8, force=False):
    root = Path(root)
    if not force and any((root / 'inputs').glob('*.jsonl')):
        raise ValueError('Data already exists. Use a different --data directory or --force explicitly.')
    for si, (split, n) in enumerate((('train', train), ('dev_iid', dev), ('dev_ood', dev))):
        rows = [make_episode(seed + 10000 * si + i, split, i) for i in range(n)]
        write_jsonl(root / 'inputs' / f'{split}.jsonl', (r[0] for r in rows))
        write_jsonl(root / 'labels' / f'{split}.jsonl', (r[1] for r in rows))
    save_json(root / 'DATA_CARD.json', {'dataset_kind': 'synthetic_fixture', 'generator_seed': seed,
              'expert_annotated': False, 'real_laboratories': 0, 'real_video_hours': 0,
              'purpose': 'Recruiting exercise and software tests, NOT wet-lab benchmark results.',
              'splits': {'train': train, 'dev_iid': dev, 'dev_ood': dev},
              'hidden_test_included': False, 'timing_units': 'seconds', 'observation_rate_hz': 1,
              'error_types': ['wrong_target', 'missing_mix']})

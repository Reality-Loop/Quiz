"""Strict, dependency-free contracts. Fail loudly rather than silently dropping bad rows."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

ACTIONS = ('background', 'attach_tip', 'aspirate', 'dispense', 'mix', 'eject_tip')
STATES = ('empty', 'loaded', 'transferred', 'mixed')
STEPS = ('background', 'S1', 'S2', 'S3', 'S4')
ERRORS = ('wrong_target', 'missing_mix')
TARGETS = ('tube_a', 'tube_b', 'unknown')
ACTION_TO_STEP = dict(zip(ACTIONS, ('background', 'S1', 'S2', 'S2', 'S3', 'S4')))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite(x, name: str, minimum=0.0) -> float:
    require(isinstance(x, (int, float)) and not isinstance(x, bool), f'{name}: expected number')
    require(math.isfinite(x) and x >= minimum, f'{name}: invalid finite range')
    return float(x)


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding='utf-8') as handle:
        for i, line in enumerate(handle, 1):
            if line.strip():
                try:
                    row = json.loads(line, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
                    require(isinstance(row, dict), 'expected object')
                    rows.append(row)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f'{path}:{i}: {exc}') from exc
    require(bool(rows), f'{path}: empty input')
    return rows


def write_jsonl(path: str | Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')


def save_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')


def scores(value: dict, choices, name: str) -> None:
    require(isinstance(value, dict) and bool(value), f'{name}: nonempty object required')
    require(set(value) <= set(choices), f'{name}: unexpected category')
    for key, p in value.items():
        finite(p, f'{name}.{key}')
        require(p <= 1.0, f'{name}: scores must be in [0,1]')
    require(abs(sum(value.values()) - 1.0) <= 1e-5, f'{name}: scores must sum to one')


def validate_episode(ep: dict) -> dict[str, float]:
    require(ep.get('schema_version') == '1.0', 'unsupported episode schema')
    for field in ('episode_id', 'trial_id', 'operator_id', 'lab_id'):
        require(isinstance(ep.get(field), str) and bool(ep[field]), f'{field}: required string')
    require(ep.get('dataset_kind') in ('synthetic_fixture', 'real_video'), 'dataset_kind required')
    duration = finite(ep.get('duration_s'), 'duration_s')
    require(duration > 0, 'duration must be positive')
    protocol = ep.get('protocol', {})
    require(protocol.get('target_id') in TARGETS[:2], 'protocol.target_id invalid')
    require(isinstance(protocol.get('mix_required'), bool), 'protocol.mix_required must be boolean')
    require(bool(protocol.get('steps')), 'protocol.steps required')
    require([s.get('id') for s in protocol['steps']] == ['S1', 'S2', 'S3', 'S4'], 'invalid step IDs')
    obs = ep.get('observations')
    require(isinstance(obs, list) and bool(obs), 'observations required')
    evidence = {}
    prev = -1.0
    for o in obs:
        t = finite(o.get('t'), 'observation.t')
        require(prev < t < duration, 'timestamps must be strictly increasing and below duration')
        prev = t
        require(isinstance(o.get('views'), list), 'views must be a list; empty is allowed')
        cameras = set()
        for v in o['views']:
            require(isinstance(v.get('camera_id'), str) and bool(v['camera_id']), 'camera_id required')
            require(v['camera_id'] not in cameras, 'duplicate camera in a timestep')
            cameras.add(v['camera_id'])
            source_t = finite(v.get('source_t'), 'view.source_t')
            require(source_t <= t, 'future view is forbidden')
            eid = v.get('evidence_id')
            require(isinstance(eid, str) and bool(eid) and eid not in evidence, 'evidence IDs must be unique')
            require(isinstance(v.get('visible'), bool), 'visible must be boolean')
            evidence[eid] = source_t
            scores(v.get('action_scores'), (*ACTIONS, 'unknown'), 'action_scores')
            scores(v.get('state_scores'), (*STATES, 'unknown'), 'state_scores')
            scores(v.get('target_scores'), TARGETS, 'target_scores')
    require(obs[0]['t'] == 0, 'first observation must be at t=0')
    return evidence


def validate_labels(ep: dict, gt: dict) -> None:
    require(gt.get('episode_id') == ep['episode_id'], 'label episode mismatch')
    require(gt.get('annotation_source') in ('synthetic_program', 'expert_reviewed'), 'annotation_source required')
    if ep['dataset_kind'] == 'real_video':
        require(gt['annotation_source'] == 'expert_reviewed', 'real GT must be expert reviewed, not model output')
        require(bool(gt.get('reviewer_id')) and bool(gt.get('review_record')), 'real GT review record required')
    frames = gt.get('frames', [])
    require([f.get('t') for f in frames] == [o['t'] for o in ep['observations']], 'GT frame timestamps mismatch')
    for f in frames:
        require(f.get('action') in (*ACTIONS, 'unknown'), 'invalid GT action')
        require(f.get('state') in (*STATES, 'unknown'), 'invalid GT state')
        require(f.get('step_id') in (*STEPS, 'unknown'), 'invalid GT step')
    events = gt.get('events')
    require(isinstance(events, list), 'events list required')
    for event in events:
        require(event.get('type') in ERRORS and event.get('scope_id') in TARGETS[:2], 'invalid GT event identity')
        occur = finite(event.get('t_occur'), 't_occur')
        end = finite(event.get('t_end'), 't_end')
        require(occur <= end <= ep['duration_s'], 'invalid event interval')
        for key in ('t_detectable', 't_pnr'):
            require(key in event, f'{key} must be explicit, null is allowed')
            if event[key] is not None:
                finite(event[key], key)
                require(event[key] <= ep['duration_s'], f'{key} out of recording')
        if event['t_detectable'] is not None:
            require(event['t_detectable'] <= end, 't_detectable after t_end')
        if event['t_pnr'] is not None:
            require(event['t_pnr'] >= occur, 't_pnr before occurrence')
        require(event.get('severity') in ('low', 'medium', 'high', 'unknown'), 'invalid severity')
        require(isinstance(event.get('evidence_ids'), list), 'event evidence required')
        if ep['dataset_kind'] == 'real_video':
            require(bool(event['evidence_ids']), 'real event needs expert evidence')
            if event['t_pnr'] is not None:
                require(bool(event.get('pnr_rationale')), 'real PNR needs an expert rationale')
        available = {v['evidence_id'] for o in ep['observations'] for v in o['views']}
        require(set(event['evidence_ids']) <= available, 'unknown GT evidence reference')


def validate_decision(d: dict, current_t: float, evidence: dict[str, float]) -> None:
    require(isinstance(d, dict), 'agent must return an object')
    require(set(d) == {'action', 'state', 'step_id', 'evidence_ids', 'alerts'}, 'unexpected/missing decision fields')
    require(d['action'] in (*ACTIONS, 'unknown'), 'invalid predicted action')
    require(d['state'] in (*STATES, 'unknown'), 'invalid predicted state')
    require(d['step_id'] in (*STEPS, 'unknown'), 'invalid predicted step')
    require(isinstance(d['alerts'], list), 'alerts must be a list')
    evidence_lists = [d['evidence_ids']]
    for alert in d['alerts']:
        require(set(alert) == {'type', 'scope_id', 'evidence_ids'}, 'alert fields invalid; runner owns alarm time')
        require(alert['type'] in ERRORS and alert['scope_id'] in TARGETS[:2], 'invalid alert identity')
        require(bool(alert['evidence_ids']), 'alarm without evidence is forbidden')
        evidence_lists.append(alert['evidence_ids'])
    for refs in evidence_lists:
        require(isinstance(refs, list) and all(isinstance(r, str) for r in refs), 'evidence must be a string list')
        require(len(set(refs)) == len(refs), 'duplicate evidence within decision')
        for ref in refs:
            require(ref in evidence and evidence[ref] <= current_t, 'unknown/future evidence')
    if any(d[k] not in ('unknown', 'background') for k in ('action', 'state', 'step_id')):
        require(bool(d['evidence_ids']), 'non-abstaining decision needs evidence')


def validate_prediction(ep: dict, pred: dict) -> None:
    require(pred.get('episode_id') == ep['episode_id'], 'prediction episode mismatch')
    require(pred.get('timing_mode') == 'video_time_fixed_delay', 'unsupported timing_mode')
    delay = finite(pred.get('fixed_delay_s'), 'fixed_delay_s')
    require(pred.get('view_policy') in ('all', 'first'), 'view_policy required')
    frames = pred.get('decisions', [])
    require(len(frames) == len(ep['observations']), 'prediction timestep count mismatch')
    evidence = {}
    for o, row in zip(ep['observations'], frames):
        require(row.get('t_observed') == o['t'], 'prediction timestamp mismatch')
        finite(row.get('t_emitted'), 't_emitted')
        require(abs(row['t_emitted'] - (o['t'] + delay)) < 1e-7, 'alarm time cannot be backdated or changed')
        finite(row.get('processing_seconds'), 'processing_seconds')
        evidence.update({v['evidence_id']: v['source_t'] for v in o['views']
                         if pred['view_policy'] == 'all' or v['camera_id'] == 'cam_1'})
        validate_decision(row.get('decision'), o['t'], evidence)


def unique_by_id(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        key = row.get('episode_id')
        require(isinstance(key, str) and key not in result, 'missing/duplicate episode_id')
        result[key] = row
    return result


def audit_splits(root: Path) -> dict:
    trials, digests, operators, labs = {}, {}, {}, {}
    counts = {}
    for path in sorted((root / 'inputs').glob('*.jsonl')):
        eps = unique_by_id(load_jsonl(path))
        labpath = root / 'labels' / path.name
        labels = unique_by_id(load_jsonl(labpath)) if labpath.exists() else None
        if labels is not None:
            require(set(eps) == set(labels), 'labels and inputs must cover identical episodes')
        counts[path.stem] = len(eps)
        operators[path.stem], labs[path.stem] = set(), set()
        for ep in eps.values():
            validate_episode(ep)
            if labels is not None:
                validate_labels(ep, labels[ep['episode_id']])
            trial = ep['trial_id']
            require(trial not in trials, f'cross-split/duplicate trial: {trial}')
            trials[trial] = path.stem
            # IDs are removed so renamed exact copies are still detected.
            content = [{"t": o['t'], "views": [{k: v for k, v in view.items() if k not in ('evidence_id', 'camera_id')}
                        for view in o['views']]} for o in ep['observations']]
            digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
            require(digest not in digests, 'duplicated observation content across episodes/splits')
            digests[digest] = path.stem
            operators[path.stem].add(ep['operator_id'])
            labs[path.stem].add(ep['lab_id'])
    require(bool(counts), 'no input files found')
    if 'train' in labs and 'dev_ood' in labs:
        require(not (labs['train'] & labs['dev_ood']), 'unseen-lab split overlaps training labs')
    return {'episodes_per_split': counts, 'trial_overlap': False, 'exact_content_overlap': False,
            'labs_per_split': {k: sorted(v) for k, v in labs.items()},
            'operators_per_split': {k: sorted(v) for k, v in operators.items()},
            'notice': 'Synthetic site/operator IDs are software fixtures, not evidence of real lab generalization.'}

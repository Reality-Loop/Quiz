"""Transparent metrics: no LLM judges, no credit for duplicate alarms, explicit denominators."""
from __future__ import annotations
import statistics
from .schema import ACTIONS, STEPS, validate_episode, validate_labels, validate_prediction, unique_by_id, require


def ratio(n, d):
    return n / d if d else None


def prf(tp, fp, fn):
    # Undefined metrics stay null. In particular, an empty-negative set does not score 1.0.
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': ratio(tp, tp + fp),
            'recall': ratio(tp, tp + fn), 'f1': ratio(2 * tp, 2 * tp + fp + fn)}


def temporal_iou(a, b):
    intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return intersection / union if union > 0 else 0.0


def maximum_matching(edges: list[list[int]], n_right: int) -> list[tuple[int, int]]:
    """Maximum-cardinality bipartite matching; no greedy-IoU matching shortcut."""
    matched = [-1] * n_right
    def visit(left, visited):
        for right in edges[left]:
            if right in visited:
                continue
            visited.add(right)
            if matched[right] < 0 or visit(matched[right], visited):
                matched[right] = left
                return True
        return False
    for left in range(len(edges)):
        visit(left, set())
    return [(left, right) for right, left in enumerate(matched) if left >= 0]


def segments(frames: list[dict], duration: float, key: str):
    result = []
    for i, f in enumerate(frames):
        end = frames[i + 1]['t'] if i + 1 < len(frames) else duration
        if result and result[-1]['label'] == f[key] and result[-1]['end'] == f['t']:
            result[-1]['end'] = end
        else:
            result.append({'label': f[key], 'start': f['t'], 'end': end})
    return [s for s in result if s['label'] not in ('background', 'unknown')]


def segment_counts(pred_frames, gt_frames, duration, threshold=.5):
    pred = segments(pred_frames, duration, 'step_id')
    truth = segments(gt_frames, duration, 'step_id')
    edges = [[j for j, g in enumerate(truth)
              if p['label'] == g['label'] and temporal_iou((p['start'], p['end']), (g['start'], g['end'])) >= threshold]
             for p in pred]
    tp = len(maximum_matching(edges, len(truth)))
    return tp, len(pred) - tp, len(truth) - tp


def event_counts(alerts: list[dict], events: list[dict], duration: float) -> dict:
    alerts = sorted(alerts, key=lambda a: a['t_emitted'])
    def compatible(a, g):
        start = g['t_detectable'] if g['t_detectable'] is not None else g['t_occur']
        return (a['type'] == g['type'] and a['scope_id'] == g['scope_id']
                and start <= a['t_emitted'] <= g['t_end'])
    edges = [[j for j, g in enumerate(events) if compatible(a, g)] for a in alerts]
    pairs = maximum_matching(edges, len(events))
    tp = len(pairs)
    opportunity_ids = [j for j, g in enumerate(events) if g['t_detectable'] is not None
                       and g['t_pnr'] is not None and g['t_detectable'] < g['t_pnr']]
    timely_edges = [[k for k, j in enumerate(opportunity_ids) if compatible(a, events[j])
                    and a['t_emitted'] < events[j]['t_pnr']] for a in alerts]
    timely = maximum_matching(timely_edges, len(opportunity_ids))
    anticipation_ids = [j for j, g in enumerate(events) if g['t_detectable'] is not None
                        and g['t_detectable'] < g['t_occur']]
    early_edges = [[k for k, j in enumerate(anticipation_ids) if compatible(a, events[j])
                   and a['t_emitted'] < events[j]['t_occur']] for a in alerts]
    early = maximum_matching(early_edges, len(anticipation_ids))
    delays = [alerts[i]['t_emitted'] - events[j]['t_detectable'] for i, j in pairs
              if events[j]['t_detectable'] is not None]
    leads = [events[opportunity_ids[k]]['t_pnr'] - alerts[i]['t_emitted'] for i, k in timely]
    return {'tp': tp, 'fp': len(alerts) - tp, 'fn': len(events) - tp,
            'opportunities': len(opportunity_ids), 'timely_tp': len(timely),
            'anticipation_opportunities': len(anticipation_ids), 'anticipation_tp': len(early),
            'unknown_pnr': sum(g['t_pnr'] is None for g in events),
            'unknown_detectable': sum(g['t_detectable'] is None for g in events),
            'no_intervention_window': sum(g['t_detectable'] is not None and g['t_pnr'] is not None
                                          and g['t_detectable'] >= g['t_pnr'] for g in events),
            'delays': delays, 'lead_times': leads, 'duration_s': duration}


def evaluate(episodes: list[dict], labels: list[dict], predictions: list[dict]) -> dict:
    inputs, gt, preds = map(unique_by_id, (episodes, labels, predictions))
    require(set(inputs) == set(gt) == set(preds), 'input/GT/prediction episode IDs must match exactly')
    by_episode, all_gt_actions, all_pred_actions = [], [], []
    action_ignored = 0
    action_correct = action_total = state_correct = state_total = state_answered = 0
    step_correct = step_total = 0
    seg = [0, 0, 0]
    raw_event, latencies = [], []
    for eid, ep in inputs.items():
        truth, pred = gt[eid], preds[eid]
        validate_episode(ep)
        validate_labels(ep, truth)
        validate_prediction(ep, pred)
        pf = []
        alerts = []
        local_step_correct = local_step_total = 0
        for f, d in zip(truth['frames'], pred['decisions']):
            p = d['decision']
            pf.append({'t': d['t_observed'], 'step_id': p['step_id']})
            if f['action'] != 'unknown':
                all_gt_actions.append(f['action'])
                all_pred_actions.append(p['action'])
                action_correct += f['action'] == p['action']
                action_total += 1
            else:
                action_ignored += 1
            if f['state'] != 'unknown':
                state_total += 1
                state_answered += p['state'] != 'unknown'
                state_correct += p['state'] == f['state']
            if f['step_id'] != 'unknown':
                step_total += 1
                local_step_total += 1
                step_correct += p['step_id'] == f['step_id']
                local_step_correct += p['step_id'] == f['step_id']
            alerts.extend({**a, 't_emitted': d['t_emitted']} for a in p['alerts'])
            latencies.append(d['processing_seconds'])
        # Unknown GT step intervals are not silently treated as real labeled boundaries.
        require(all(f['step_id'] != 'unknown' for f in truth['frames']),
                'step-segment metric requires complete step labels; use a fully annotated evaluation subset')
        sc = segment_counts(pf, truth['frames'], ep['duration_s'])
        seg = [x + y for x, y in zip(seg, sc)]
        ev = event_counts(alerts, truth['events'], ep['duration_s'])
        raw_event.append(ev)
        by_episode.append({'episode_id': eid, 'step_accuracy': ratio(local_step_correct, local_step_total),
                           'step_segment_f1_50': prf(*sc)['f1'], 'event': prf(ev['tp'], ev['fp'], ev['fn']),
                           'timely_recall': ratio(ev['timely_tp'], ev['opportunities'])})
    class_metrics = {}
    for c in ACTIONS:
        tp = sum(g == c and p == c for g, p in zip(all_gt_actions, all_pred_actions))
        fp = sum(g != c and p == c for g, p in zip(all_gt_actions, all_pred_actions))
        fn = sum(g == c and p != c for g, p in zip(all_gt_actions, all_pred_actions))
        class_metrics[c] = prf(tp, fp, fn)
    # Fixed vocabulary; absent classes are reported null, then excluded from macro with support disclosed.
    defined = [m['f1'] for m in class_metrics.values() if m['f1'] is not None]
    totals = {k: sum(e[k] for e in raw_event) for k in
              ('tp', 'fp', 'fn', 'opportunities', 'timely_tp', 'anticipation_opportunities', 'anticipation_tp',
               'unknown_pnr', 'unknown_detectable', 'no_intervention_window', 'duration_s')}
    delays = [d for e in raw_event for d in e['delays']]
    leads = [d for e in raw_event for d in e['lead_times']]
    kinds = sorted({ep['dataset_kind'] for ep in episodes})
    return {'schema_version': '1.0', 'dataset_kinds': kinds,
            'scientific_benchmark_valid': False,
            'real_expert_provenance_declared': kinds == ['real_video'],
            'validity_notice': ('Synthetic recruiting fixtures ONLY. Not wet-lab performance or real generalization.'
                                if 'synthetic_fixture' in kinds else
                                'Expert provenance is declared, not independently certified by this software.'),
            'num_episodes': len(inputs), 'num_timestamps': len(latencies),
            'action_accuracy': ratio(action_correct, action_total),
            'action_macro_f1': statistics.mean(defined) if defined else None,
            'action_macro_defined_classes': len(defined), 'action_by_class': class_metrics,
            'action_unknown_gt_ignored': action_ignored,
            'step_accuracy': ratio(step_correct, step_total), 'step_segment_f1_50': prf(*seg),
            'state_accuracy_on_known_gt': ratio(state_correct, state_total),
            'state_coverage_on_known_gt': ratio(state_answered, state_total),
            'event': {**prf(totals['tp'], totals['fp'], totals['fn']),
                      'false_alarms_per_hour': ratio(totals['fp'], totals['duration_s'] / 3600),
                      'timely_recall': ratio(totals['timely_tp'], totals['opportunities']),
                      'timely_tp': totals['timely_tp'], 'intervention_opportunities': totals['opportunities'],
                      'anticipation_recall': ratio(totals['anticipation_tp'], totals['anticipation_opportunities']),
                      'anticipation_opportunities': totals['anticipation_opportunities'],
                      'unknown_pnr_excluded': totals['unknown_pnr'],
                      'unknown_detectable_excluded': totals['unknown_detectable'],
                      'no_intervention_window': totals['no_intervention_window'],
                      'detection_delay_mean_s_on_matched': statistics.mean(delays) if delays else None,
                      'lead_time_mean_s_on_timely': statistics.mean(leads) if leads else None},
            'runtime': {'step_processing_p50_ms': statistics.median(latencies) * 1000 if latencies else None,
                        'step_processing_max_ms': max(latencies) * 1000 if latencies else None,
                        'timing_mode': 'video_time_fixed_delay; wall clock reported separately; no real-time claim'},
            'per_episode': by_episode}

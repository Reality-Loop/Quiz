from __future__ import annotations
import copy
from pathlib import Path
import tempfile
import unittest
from wetlab_challenge.fixtures import make_episode, generate
from wetlab_challenge.schema import (validate_episode, validate_labels, validate_prediction,
                                     validate_decision, audit_splits, load_jsonl, write_jsonl, unique_by_id)
from wetlab_challenge.runner import run_episode, load_agent
from wetlab_challenge.metrics import (temporal_iou, maximum_matching, event_counts, prf, evaluate, segments)


def event(td=2.0, pnr=5.0, occur=3.0, end=10.0):
    return {'type': 'wrong_target', 'scope_id': 'tube_b', 't_detectable': td,
            't_pnr': pnr, 't_occur': occur, 't_end': end}


def alert(t, scope='tube_b'):
    return {'type': 'wrong_target', 'scope_id': scope, 't_emitted': t}


class MetricTests(unittest.TestCase):
    def test_iou_overlap(self):
        self.assertAlmostEqual(temporal_iou((0, 3), (1, 4)), .5)

    def test_iou_touching_intervals(self):
        self.assertEqual(temporal_iou((0, 1), (1, 2)), 0)

    def test_iou_zero_length(self):
        self.assertEqual(temporal_iou((1, 1), (1, 1)), 0)

    def test_max_matching_not_greedy(self):
        self.assertEqual(len(maximum_matching([[0, 1], [0]], 2)), 2)

    def test_empty_metrics_are_undefined_not_perfect(self):
        self.assertIsNone(prf(0, 0, 0)['f1'])

    def test_missed_event_f1_zero(self):
        self.assertEqual(prf(0, 0, 2)['f1'], 0)

    def test_duplicate_alert_is_false_positive(self):
        result = event_counts([alert(3), alert(4)], [event()], 10)
        self.assertEqual((result['tp'], result['fp'], result['fn']), (1, 1, 0))

    def test_too_early_without_observable_evidence_is_fp(self):
        result = event_counts([alert(1)], [event()], 10)
        self.assertEqual((result['tp'], result['fp'], result['fn']), (0, 1, 1))

    def test_alarm_at_pnr_is_not_timely(self):
        result = event_counts([alert(5)], [event()], 10)
        self.assertEqual(result['tp'], 1)
        self.assertEqual(result['timely_tp'], 0)

    def test_alarm_before_occurrence_is_anticipation(self):
        result = event_counts([alert(2)], [event()], 10)
        self.assertEqual(result['anticipation_tp'], 1)
        self.assertEqual(result['timely_tp'], 1)

    def test_null_pnr_is_excluded(self):
        result = event_counts([alert(3)], [event(pnr=None)], 10)
        self.assertEqual(result['opportunities'], 0)
        self.assertEqual(result['unknown_pnr'], 1)
        self.assertEqual(result['tp'], 1)

    def test_null_detectable_is_excluded(self):
        result = event_counts([alert(3)], [event(td=None)], 10)
        self.assertEqual(result['opportunities'], 0)
        self.assertEqual(result['unknown_detectable'], 1)

    def test_no_opportunity_not_unfairly_in_denominator(self):
        result = event_counts([alert(6)], [event(td=6)], 10)
        self.assertEqual(result['opportunities'], 0)
        self.assertEqual(result['no_intervention_window'], 1)

    def test_wrong_entity_does_not_match(self):
        result = event_counts([alert(3, 'tube_a')], [event()], 10)
        self.assertEqual(result['tp'], 0)

    def test_alert_after_episode_window_is_fp(self):
        self.assertEqual(event_counts([alert(11)], [event()], 10)['fp'], 1)

    def test_no_error_episode_penalizes_false_alarm(self):
        self.assertEqual(event_counts([alert(3)], [], 10)['fp'], 1)

    def test_one_alarm_cannot_detect_two_events(self):
        self.assertEqual(event_counts([alert(3)], [event(), event()], 10)['tp'], 1)

    def test_repeated_step_segments_are_not_collapsed_across_gaps(self):
        frames = [{'t': 0, 'step_id': 'S1'}, {'t': 1, 'step_id': 'background'}, {'t': 2, 'step_id': 'S1'}]
        self.assertEqual(len(segments(frames, 3, 'step_id')), 2)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.ep, self.gt = make_episode(100, 'dev_iid', 1)

    def test_fixture_and_labels_valid(self):
        validate_episode(self.ep)
        validate_labels(self.ep, self.gt)

    def test_duplicate_timestamp_fails(self):
        self.ep['observations'][1]['t'] = 0
        with self.assertRaises(ValueError):
            validate_episode(self.ep)

    def test_future_camera_timestamp_fails(self):
        self.ep['observations'][0]['views'][0]['source_t'] = 999
        with self.assertRaises(ValueError):
            validate_episode(self.ep)

    def test_nan_score_fails(self):
        self.ep['observations'][0]['views'][0]['action_scores']['background'] = float('nan')
        with self.assertRaises(ValueError):
            validate_episode(self.ep)

    def test_missing_explicit_null_pnr_fails(self):
        del self.gt['events'][0]['t_pnr']
        with self.assertRaises(ValueError):
            validate_labels(self.ep, self.gt)

    def test_real_gt_cannot_be_synthetic(self):
        self.ep['dataset_kind'] = 'real_video'
        with self.assertRaises(ValueError):
            validate_labels(self.ep, self.gt)

    def test_duplicate_episode_ids_fail(self):
        with self.assertRaises(ValueError):
            unique_by_id([self.ep, self.ep])

    def test_missing_predictions_fail(self):
        with self.assertRaises(ValueError):
            evaluate([self.ep], [self.gt], [])

    def test_backdated_alarm_fails(self):
        pred = run_episode(self.ep, load_agent('baseline'))
        pred['decisions'][1]['t_emitted'] = 0
        with self.assertRaises(ValueError):
            validate_prediction(self.ep, pred)

    def test_prefix_api_has_no_future_or_labels(self):
        testcase = self
        class RecordingAgent:
            def reset(self, context):
                testcase.assertEqual(set(context), {'episode_id', 'protocol'})
            def update(self, observation):
                testcase.assertEqual(set(observation), {'t', 'views'})
                return {'action': 'unknown', 'state': 'unknown', 'step_id': 'unknown',
                        'evidence_ids': [], 'alerts': []}
        run_episode(self.ep, RecordingAgent())

    def test_future_evidence_fails(self):
        decision = {'action': 'dispense', 'state': 'loaded', 'step_id': 'S2',
                    'evidence_ids': ['future'], 'alerts': []}
        with self.assertRaises(ValueError):
            validate_decision(decision, 0, {'future': 1})

    def test_candidate_cannot_supply_alarm_timestamp(self):
        decision = {'action': 'unknown', 'state': 'unknown', 'step_id': 'unknown', 'evidence_ids': [],
                    'alerts': [{'type': 'wrong_target', 'scope_id': 'tube_b',
                                'evidence_ids': ['x'], 't_emitted': 0}]}
        with self.assertRaises(ValueError):
            validate_decision(decision, 0, {'x': 0})

    def test_prefix_invariance(self):
        full = run_episode(self.ep, load_agent('baseline'))
        prefix = copy.deepcopy(self.ep)
        prefix['observations'] = prefix['observations'][:8]
        prefix['duration_s'] = 8.0
        short = run_episode(prefix, load_agent('baseline'))
        self.assertEqual([d['decision'] for d in full['decisions'][:8]],
                         [d['decision'] for d in short['decisions']])

    def test_submission_contract_is_runnable(self):
        prediction = run_episode(self.ep, load_agent('submission'))
        validate_prediction(self.ep, prediction)

    def test_fixed_latency_is_applied_by_runner(self):
        pred = run_episode(self.ep, load_agent('baseline'), delay_s=2.5)
        self.assertEqual(pred['decisions'][0]['t_emitted'], 2.5)
        validate_prediction(self.ep, pred)

    def test_baseline_metrics_finite_and_synthetic_flag(self):
        pred = run_episode(self.ep, load_agent('baseline'))
        result = evaluate([self.ep], [self.gt], [pred])
        self.assertFalse(result['scientific_benchmark_valid'])
        self.assertGreaterEqual(result['action_accuracy'], 0)
        self.assertLessEqual(result['action_accuracy'], 1)

    def test_first_view_mode_rejects_second_view_evidence(self):
        pred = run_episode(self.ep, load_agent('baseline'), view_policy='first')
        for o, row in zip(self.ep['observations'], pred['decisions']):
            second = [v for v in o['views'] if v['camera_id'] == 'cam_2']
            if second:
                row['decision']['evidence_ids'] = [second[0]['evidence_id']]
                break
        with self.assertRaises(ValueError):
            validate_prediction(self.ep, pred)

    def test_mutating_agent_cannot_change_inputs(self):
        original = copy.deepcopy(self.ep)
        class MutatingAgent:
            def reset(self, context):
                context['protocol']['target_id'] = 'tube_b'
            def update(self, observation):
                observation['views'].clear()
                return {'action': 'unknown', 'state': 'unknown', 'step_id': 'unknown',
                        'evidence_ids': [], 'alerts': []}
        run_episode(self.ep, MutatingAgent())
        self.assertEqual(self.ep, original)


class DatasetTests(unittest.TestCase):
    def test_generate_audit_and_refuse_unintentional_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            generate(root, train=4, dev=4)
            result = audit_splits(root)
            self.assertEqual(result['episodes_per_split']['train'], 4)
            with self.assertRaises(ValueError):
                generate(root)

    def test_cross_split_trial_leak_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            generate(root, train=4, dev=4)
            train = load_jsonl(root / 'inputs/train.jsonl')
            dev = load_jsonl(root / 'inputs/dev_iid.jsonl')
            dev[0]['trial_id'] = train[0]['trial_id']
            write_jsonl(root / 'inputs/dev_iid.jsonl', dev)
            with self.assertRaises(ValueError):
                audit_splits(root)

    def test_invalid_json_nan_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'bad.jsonl'
            p.write_text('{"a":NaN}\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_jsonl(p)

    def test_renamed_exact_duplicate_content_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            generate(root, train=4, dev=4)
            (root / 'labels/train.jsonl').unlink()
            rows = load_jsonl(root / 'inputs/train.jsonl')
            clone = copy.deepcopy(rows[0])
            clone['episode_id'] = 'renamed'
            clone['trial_id'] = 'renamed_trial'
            for o in clone['observations']:
                for v in o['views']:
                    v['evidence_id'] = 'renamed:' + v['evidence_id']
            rows.append(clone)
            write_jsonl(root / 'inputs/train.jsonl', rows)
            with self.assertRaises(ValueError):
                audit_splits(root)


if __name__ == '__main__':
    unittest.main()

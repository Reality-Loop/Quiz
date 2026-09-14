"""Deliberately simple single-view, per-frame baseline. No labels or future input."""
from .schema import ACTION_TO_STEP


class BaselineAgent:
    def reset(self, context: dict) -> None:
        self.protocol = context['protocol']

    def update(self, observation: dict) -> dict:
        views = [v for v in observation['views'] if v['visible']]
        if not views:
            return {'action': 'unknown', 'state': 'unknown', 'step_id': 'unknown',
                    'evidence_ids': [], 'alerts': []}
        view = views[0]
        action = max(view['action_scores'], key=view['action_scores'].get)
        state = max(view['state_scores'], key=view['state_scores'].get)
        target = max(view['target_scores'], key=view['target_scores'].get)
        refs = [view['evidence_id']]
        alerts = []
        if action == 'dispense' and target not in ('unknown', self.protocol['target_id']):
            alerts.append({'type': 'wrong_target', 'scope_id': target, 'evidence_ids': refs})
        # Intentionally no temporal memory, anticipation, missing-step reasoning, or incident deduplication.
        return {'action': action, 'state': state, 'step_id': ACTION_TO_STEP.get(action, 'unknown'),
                'evidence_ids': refs, 'alerts': alerts}

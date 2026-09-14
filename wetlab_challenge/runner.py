"""Prefix-only API runner. It is a contract, NOT a security sandbox for hostile code."""
from __future__ import annotations
import copy
import importlib
import time
from .schema import validate_episode, validate_decision, finite, require


def load_agent(spec: str):
    aliases = {'baseline': 'wetlab_challenge.baseline:BaselineAgent', 'submission': 'submission.agent:Agent'}
    spec = aliases.get(spec, spec)
    require(':' in spec, 'agent must be baseline, submission or module:Class')
    module, name = spec.rsplit(':', 1)
    agent = getattr(importlib.import_module(module), name)()
    require(callable(getattr(agent, 'reset', None)) and callable(getattr(agent, 'update', None)),
            'agent must implement reset and update')
    return agent


def run_episode(ep: dict, agent, delay_s=0.0, view_policy='all') -> dict:
    validate_episode(ep)
    finite(delay_s, 'delay_s')
    require(view_policy in ('all', 'first'), 'unsupported view policy')
    # DO NOT pass lab/operator/split IDs, the complete episode, GT, or recording duration to the agent.
    agent.reset(copy.deepcopy({'episode_id': ep['episode_id'], 'protocol': ep['protocol']}))
    decisions, available_evidence = [], {}
    for source in ep['observations']:
        obs = copy.deepcopy(source)
        if view_policy == 'first':
            obs['views'] = [v for v in obs['views'] if v['camera_id'] == 'cam_1']
        available_evidence.update({v['evidence_id']: v['source_t'] for v in obs['views']})
        started = time.perf_counter()
        decision = copy.deepcopy(agent.update(obs))
        elapsed = time.perf_counter() - started
        validate_decision(decision, source['t'], available_evidence)
        decisions.append({'t_observed': source['t'], 't_emitted': source['t'] + delay_s,
                          'processing_seconds': elapsed, 'decision': decision})
    return {'episode_id': ep['episode_id'], 'timing_mode': 'video_time_fixed_delay',
            'fixed_delay_s': delay_s, 'view_policy': view_policy, 'decisions': decisions}

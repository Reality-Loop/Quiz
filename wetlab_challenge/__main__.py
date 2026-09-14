from __future__ import annotations
import argparse
import platform
import sys
from pathlib import Path
from .fixtures import generate
from .schema import load_jsonl, write_jsonl, save_json, audit_splits
from .runner import load_agent, run_episode
from .metrics import evaluate


def run(data, split, agent, out, delay, views):
    episodes = load_jsonl(data / 'inputs' / f'{split}.jsonl')
    predictions = [run_episode(ep, load_agent(agent), delay, views) for ep in episodes]
    write_jsonl(out, predictions)
    return episodes, predictions


def main():
    p = argparse.ArgumentParser(description='RealityLoop causal observation-stream challenge (synthetic fixtures).')
    sub = p.add_subparsers(dest='command', required=True)
    g = sub.add_parser('generate', help='Generate synthetic fixtures; never downloads real data')
    g.add_argument('--data', type=Path, default=Path('data'))
    g.add_argument('--seed', type=int, default=20260914)
    g.add_argument('--force', action='store_true')
    a = sub.add_parser('audit')
    a.add_argument('--data', type=Path, default=Path('data'))
    a.add_argument('--out', type=Path, default=Path('outputs/audit.json'))
    for name in ('run', 'evaluate', 'demo'):
        c = sub.add_parser(name)
        c.add_argument('--data', type=Path, default=Path('data'))
        c.add_argument('--out', type=Path, default=Path('outputs') if name == 'demo' else
                       Path('outputs/predictions.jsonl' if name == 'run' else 'outputs/metrics.json'))
        if name != 'demo':
            c.add_argument('--split', default='dev_iid')
        if name != 'evaluate':
            c.add_argument('--agent', default='baseline')
            c.add_argument('--latency-s', type=float, default=0.0)
            c.add_argument('--views', choices=['all', 'first'], default='all')
        else:
            c.add_argument('--predictions', type=Path, required=True)
    args = p.parse_args()
    try:
        if args.command == 'generate':
            generate(args.data, seed=args.seed, force=args.force)
            print(f'Synthetic fixtures generated in {args.data}')
        elif args.command == 'audit':
            result = audit_splits(args.data)
            save_json(args.out, result)
            print(result['episodes_per_split'])
        elif args.command == 'run':
            run(args.data, args.split, args.agent, args.out, args.latency_s, args.views)
            print(f'Predictions: {args.out}')
        elif args.command == 'evaluate':
            result = evaluate(load_jsonl(args.data / 'inputs' / f'{args.split}.jsonl'),
                              load_jsonl(args.data / 'labels' / f'{args.split}.jsonl'),
                              load_jsonl(args.predictions))
            save_json(args.out, result)
            print(f'Metrics: {args.out}')
        else:
            audit = audit_splits(args.data)
            save_json(args.out / 'audit.json', audit)
            summary = []
            for split in ('dev_iid', 'dev_ood'):
                episodes, predictions = run(args.data, split, args.agent, args.out / f'{split}.predictions.jsonl',
                                            args.latency_s, args.views)
                result = evaluate(episodes, load_jsonl(args.data / 'labels' / f'{split}.jsonl'), predictions)
                save_json(args.out / f'{split}.metrics.json', result)
                summary.append((split, result))
            lines = ['# Smoke-test results — synthetic data only', '',
                     'No real wet-lab accuracy or generalization claim is supported by this report.', '',
                     '| Split | Action macro F1 | Step F1@0.5 | Error F1 | Timely recall |',
                     '|---|---:|---:|---:|---:|']
            def fmt(x):
                return 'N/A' if x is None else f'{x:.4f}'
            for split, r in summary:
                lines.append(f"| {split} | {fmt(r['action_macro_f1'])} | {fmt(r['step_segment_f1_50']['f1'])} | "
                             f"{fmt(r['event']['f1'])} | {fmt(r['event']['timely_recall'])} |")
            (args.out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
            save_json(args.out / 'run_environment.json', {'python': sys.version, 'platform': platform.platform(),
                      'agent': args.agent, 'views': args.views, 'fixed_delay_s': args.latency_s,
                      'gpu_required': False, 'dependencies': 'Python standard library only'})
            print('\n'.join(lines))
    except (ValueError, OSError, ImportError, AttributeError) as exc:
        p.exit(2, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()

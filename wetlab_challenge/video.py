"""Optional: bounded-memory local video -> Ollama visual observations. NEVER creates GT."""
from __future__ import annotations
import argparse
import base64
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
from .fixtures import protocol as default_protocol
from .schema import ACTIONS, STATES, TARGETS, scores, require, finite, validate_episode, write_jsonl, save_json


def response_schema():
    def score_object(choices):
        return {'type': 'object', 'properties': {c: {'type': 'number', 'minimum': 0, 'maximum': 1} for c in choices},
                'required': list(choices), 'additionalProperties': False}
    return {'type': 'object', 'properties': {
        'action_scores': score_object((*ACTIONS, 'unknown')),
        'state_scores': score_object((*STATES, 'unknown')),
        'target_scores': score_object(TARGETS)},
        'required': ['action_scores', 'state_scores', 'target_scores'], 'additionalProperties': False}


def request_ollama(images: list[str], timestamps: list[float], protocol: dict, model: str,
                   endpoint='http://127.0.0.1:11434', timeout=120.0, allow_remote=False):
    parsed = urllib.parse.urlsplit(endpoint)
    require(parsed.scheme in ('http', 'https') and bool(parsed.hostname), 'invalid Ollama endpoint')
    require(allow_remote or parsed.hostname in ('127.0.0.1', 'localhost', '::1'),
            'Remote image upload is disabled. Explicit --allow-remote is required.')
    require(not parsed.username and not parsed.password, 'do not put credentials in endpoint URLs')
    prompt = (
        'Analyze ONLY the supplied chronological image prefix. Describe the LAST image using its visual history. '
        'The SOP is an intended procedure, NOT evidence that a step happened. '
        'Do not infer invisible chemical identity, concentration, sterility or reaction completion. '
        'Output JSON containing action_scores, state_scores and target_scores. '
        'Scores in each object must sum to 1; use unknown when not visually supported. '
        'State categories are visible transfer-stage proxies: empty, loaded, transferred, mixed, unknown. '
        'Only identify tube_a/tube_b when the image visibly supports that ID; otherwise unknown. '
        f'Image timestamps in seconds: {timestamps}. Intended SOP: {json.dumps(protocol)}. '
        f'JSON schema: {json.dumps(response_schema())}'
    )
    payload = {'model': model, 'stream': False, 'format': response_schema(), 'options': {'temperature': 0},
               'messages': [{'role': 'user', 'content': prompt, 'images': images}]}
    request = urllib.request.Request(endpoint.rstrip('/') + '/api/chat',
        data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(2_000_001)
        require(len(raw) <= 2_000_000, 'model response too large')
    body = json.loads(raw)
    output = json.loads(body['message']['content'])
    require(set(output) == {'action_scores', 'state_scores', 'target_scores'}, 'model returned wrong fields')
    scores(output['action_scores'], (*ACTIONS, 'unknown'), 'action_scores')
    scores(output['state_scores'], (*STATES, 'unknown'), 'state_scores')
    scores(output['target_scores'], TARGETS, 'target_scores')
    return output


def sample_frames(video: Path, hz=1.0, max_seconds=30.0, max_side=640, assume_cfr=False):
    """Sequential decode, bounded memory, no random seek into future frames.

    Uses decoder timestamps; explicitly opt in to index/fps only for known CFR input.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ValueError('Optional dependency missing: pip install -r requirements-video.txt') from exc
    require(hz > 0 and math.isfinite(hz), 'sample rate must be positive and finite')
    require(max_seconds > 0 and math.isfinite(max_seconds), 'max_seconds must be positive and finite')
    cap = cv2.VideoCapture(str(video))
    require(cap.isOpened(), f'cannot decode video: {video}')
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if assume_cfr:
            require(fps > 0 and math.isfinite(fps), 'invalid CFR frame rate')
        previous, next_sample, index = -1.0, 0.0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = index / fps if assume_cfr else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            index += 1
            require(math.isfinite(t) and t >= 0 and t > previous,
                    'decoder PTS is invalid/non-monotonic; use --assume-cfr ONLY for verified CFR video')
            previous = t
            if t > max_seconds:
                break
            if t + 1e-7 < next_sample:
                continue
            next_sample = t + 1.0 / hz
            h, w = frame.shape[:2]
            scale = min(1.0, max_side / max(w, h))
            if scale < 1.0:
                frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
            ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            require(ok, 'JPEG encoding failed')
            yield t, encoded.tobytes()
    finally:
        cap.release()


def convert(video: Path, out: Path, model: str | None = None, endpoint='http://127.0.0.1:11434',
            hz=1.0, max_seconds=30.0, history_frames=3, dry_run=False, allow_remote=False,
            assume_cfr=False, protocol_path: Path | None = None, dataset_kind='real_video'):
    require(video.is_file(), f'missing video: {video}')
    require(history_frames >= 1, 'history_frames must be >= 1')
    require(dry_run or bool(model), '--model is required unless --dry-run')
    intended = json.loads(protocol_path.read_text(encoding='utf-8')) if protocol_path else default_protocol()
    digest = hashlib.sha256()
    with video.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    eid = digest.hexdigest()[:14]
    observations, diagnostics = [], []
    history = deque(maxlen=history_frames)
    evidence_root = out.parent / f'{out.stem}_evidence'
    evidence_root.mkdir(parents=True, exist_ok=True)
    # t is relative to the first decoded frame; no future image is passed to the model.
    t0 = None
    for source_t, jpeg in sample_frames(video, hz=hz, max_seconds=max_seconds, assume_cfr=assume_cfr):
        if t0 is None:
            t0 = source_t
        t = source_t - t0
        history.append((t, base64.b64encode(jpeg).decode('ascii')))
        evidence_id = f'{eid}:cam_1:{len(observations):04d}'
        image_path = evidence_root / f'{len(observations):04d}.jpg'
        image_path.write_bytes(jpeg)
        started = time.perf_counter()
        status, error = 'model_observation', None
        if dry_run:
            result = {'action_scores': {'unknown': 1.0}, 'state_scores': {'unknown': 1.0},
                      'target_scores': {'unknown': 1.0}}
            status = 'dry_run_no_model'
        else:
            try:
                result = request_ollama([x[1] for x in history], [x[0] for x in history], intended,
                                        model, endpoint=endpoint, allow_remote=allow_remote)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                # Preserve a failed frame as unknown, never silently fabricate a confident action.
                result = {'action_scores': {'unknown': 1.0}, 'state_scores': {'unknown': 1.0},
                          'target_scores': {'unknown': 1.0}}
                status, error = 'model_failed_unknown', str(exc)
        diagnostics.append({'t': t, 'status': status, 'error': error,
                            'model_seconds': time.perf_counter() - started,
                            'history_max_t': max(x[0] for x in history)})
        observations.append({'t': t, 'views': [{'camera_id': 'cam_1', 'source_t': t,
                             'evidence_id': evidence_id, 'evidence_path': str(image_path),
                             'visible': status == 'model_observation', **result}]})
    require(bool(observations), 'video has no decodable sampled frames')
    # Endpoint is a replay-grid convention, not a claim of exact original video duration.
    duration = observations[-1]['t'] + 1.0 / hz
    ep = {'schema_version': '1.0', 'dataset_kind': dataset_kind, 'episode_id': eid,
          'trial_id': f'trial_{eid}', 'operator_id': 'unassigned', 'lab_id': 'unassigned',
          'duration_s': duration, 'protocol': intended, 'observations': observations,
          'provenance': {'video_sha256': digest.hexdigest(), 'model': model, 'dry_run': dry_run,
                         'ground_truth_created': False, 'duration_is_replay_grid': True,
                         'timestamp_mode': 'explicit_cfr' if assume_cfr else 'decoder_pts'}}
    validate_episode(ep)
    write_jsonl(out, [ep])
    save_json(out.with_suffix('.diagnostics.json'), diagnostics)
    return ep


def main():
    p = argparse.ArgumentParser(description='Optional local video adapter. Outputs observations, never labels.')
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model')
    p.add_argument('--endpoint', default='http://127.0.0.1:11434')
    p.add_argument('--sample-hz', type=float, default=1.0)
    p.add_argument('--max-seconds', type=float, default=30.0)
    p.add_argument('--history-frames', type=int, default=3)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--allow-remote', action='store_true')
    p.add_argument('--assume-cfr', action='store_true')
    p.add_argument('--protocol', type=Path)
    p.add_argument('--dataset-kind', choices=('synthetic_fixture', 'real_video'), default='real_video')
    a = p.parse_args()
    try:
        ep = convert(a.video, a.out, a.model, a.endpoint, a.sample_hz, a.max_seconds,
                     a.history_frames, a.dry_run, a.allow_remote, a.assume_cfr, a.protocol, a.dataset_kind)
        print(f"Wrote {len(ep['observations'])} observation timestamps, 0 ground-truth labels: {a.out}")
    except (ValueError, OSError) as exc:
        p.exit(2, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()

from __future__ import annotations
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from wetlab_challenge.video import request_ollama, convert
from wetlab_challenge.fixtures import protocol


class OllamaTransportTests(unittest.TestCase):
    def test_local_http_transport_with_mock_not_a_real_model(self):
        received = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                received.append((self.path, data))
                result = {'action_scores': {'unknown': 1}, 'state_scores': {'unknown': 1},
                          'target_scores': {'unknown': 1}}
                raw = json.dumps({'message': {'content': json.dumps(result)}}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        thread.start()
        try:
            output = request_ollama(['AA=='], [0], protocol(), 'mock-test-only',
                                    endpoint=f'http://127.0.0.1:{server.server_port}')
            self.assertEqual(output['action_scores'], {'unknown': 1})
            self.assertEqual(received[0][0], '/api/chat')
            self.assertFalse(received[0][1]['stream'])
            self.assertEqual(received[0][1]['messages'][0]['images'], ['AA=='])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

    def test_remote_upload_disabled_by_default(self):
        with self.assertRaises(ValueError):
            request_ollama([], [], protocol(), 'model', endpoint='https://example.com')


@unittest.skipUnless(importlib.util.find_spec('cv2'), 'optional OpenCV not installed')
class VideoDecodeTests(unittest.TestCase):
    def test_actual_mp4_decode_and_dry_run(self):
        source = Path(__file__).resolve().parents[1] / 'examples/decoder_fixture.mp4'
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'observations.jsonl'
            ep = convert(source, out, dry_run=True, dataset_kind='synthetic_fixture')
            self.assertEqual(len(ep['observations']), 4)
            self.assertEqual([o['t'] for o in ep['observations']], [0, 1, 2, 3])
            self.assertFalse(ep['provenance']['ground_truth_created'])
            self.assertTrue(out.exists())
            self.assertTrue(out.with_suffix('.diagnostics.json').exists())
            self.assertFalse((Path(d) / 'labels').exists())

    def test_missing_video_reports_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                convert(Path(d) / 'missing.mp4', Path(d) / 'out.jsonl', dry_run=True)


if __name__ == '__main__':
    unittest.main()

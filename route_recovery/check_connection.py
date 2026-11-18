"""Run on Laptop 2 to check Laptop 1, CORS and optional real route reception."""
import argparse
import json
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api', required=True, help='For example http://192.168.1.10:8000')
    parser.add_argument('--origin', default='http://localhost:8080')
    parser.add_argument('--trip', help='Optional student-test trip ID from /trips')
    parser.add_argument('--output', default='reports/laptop2_connection.json')
    args = parser.parse_args()
    base = args.api.rstrip('/')
    if urlparse(base).scheme not in ('http', 'https'):
        parser.error('--api must start with http:// or https://')
    report = {'api': base, 'ui_origin': args.origin, 'checks': [], 'ok': False}

    def request(path, method='GET', payload=None, extra=None):
        headers = {'Origin': args.origin, **(extra or {})}
        body = None if payload is None else json.dumps(payload).encode()
        if body is not None:
            headers['Content-Type'] = 'application/json'
        req = Request(base + path, data=body, headers=headers, method=method)
        with urlopen(req, timeout=180 if method == 'POST' else 15) as response:
            if response.headers.get('Access-Control-Allow-Origin') != args.origin:
                raise ValueError('API is reachable but CORS does not allow this UI origin')
            data = response.read()
        report['checks'].append({'path': path, 'method': method, 'received_bytes': len(data)})
        print(method, path, 'OK', len(data), 'bytes received', flush=True)
        return json.loads(data) if method != 'OPTIONS' else None

    try:
        health = request('/health')
        if not health.get('default_model_available'):
            raise ValueError('Selected model is missing on Laptop 1; download its artifact')
        report['model'] = health['default_model']
        inventory = request('/trips')
        report['student_test_trips'] = len(inventory['student_test'])
        request('/visualize', method='OPTIONS', extra={
            'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'content-type'})
        if args.trip:
            if args.trip not in inventory['student_test']:
                raise ValueError('Trip is absent from /trips student_test inventory')
            data = request('/visualize', method='POST', payload={
                'partition': 'student_test', 'trip_id': args.trip,
                'diagnose': True, 'evaluate': True})
            result = data['result']
            if not data['map_html'] or result['model'] != health['default_model']:
                raise ValueError('Response did not contain selected-model route/map')
            report['received_route'] = {key: result[key] for key in (
                'trip_id', 'model', 'elapsed_seconds', 'matched_observations',
                'unmatched_observations', 'section_count', 'evaluation')}
        report['ok'] = True
    except (URLError, OSError, ValueError, KeyError) as exc:
        report['error'] = str(exc)
        print('Connection check FAILED:', exc, flush=True)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Evidence saved:', target, flush=True)
    raise SystemExit(0 if report['ok'] else 1)


if __name__ == '__main__':
    main()

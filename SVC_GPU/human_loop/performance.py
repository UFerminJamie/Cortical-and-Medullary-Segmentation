"""Bounded request observations; no query strings, tokens, or patient names."""
from collections import deque
import threading


class RequestMetrics:
    def __init__(self):
        self._lock = threading.RLock()
        self._items = {}

    def observe(self, path, elapsed_ms, status):
        if path.startswith('/api/wsi/'):
            name = 'wsi'
        elif path.endswith('/corrections') or path.endswith('/undo') or path.endswith('/review'):
            name = 'save'
        elif path.endswith('/view'):
            name = 'view'
        elif path.endswith('/history'):
            name = 'history'
        elif path == '/api/slides':
            name = 'queue'
        elif path.startswith('/api/datasets'):
            name = 'datasets'
        else:
            name = 'other'
        with self._lock:
            item = self._items.setdefault(name, {'count': 0, 'errors': 0, 'not_modified': 0, 'samples': deque(maxlen=256)})
            item['count'] += 1;item['errors'] += int(status >= 400);item['not_modified'] += int(status == 304)
            item['samples'].append(elapsed_ms)

    def snapshot(self):
        result = {}
        with self._lock:
            for name, item in self._items.items():
                values = sorted(item['samples'])
                result[name] = {key: item[key] for key in ('count', 'errors', 'not_modified')}
                result[name].update(recent_samples=len(values), p50_ms=values[(len(values)-1)//2],
                                    p95_ms=values[min(len(values)-1, int(len(values)*.95))])
        return result

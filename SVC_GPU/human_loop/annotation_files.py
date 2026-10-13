"""Schema-preserving working annotations; independent of tile correction events."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import re
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

from shapely.geometry import Polygon, GeometryCollection, shape, mapping
from shapely.ops import unary_union
from shapely.validation import explain_validity

from .storage import Conflict, atomic_json, now, read_json

SOURCE_NAMES = ('cortical_gt.json', 'medullary_gt.json', 'Tissue_qc.json')
DERIVED_NAMES = ('cortical.json', 'medullary.json')
ALGORITHM = 'vector-medulla-priority-v1'


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else None


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_. ();-]{1,160}', value) or value in ('.', '..'):
        raise ValueError('Invalid sample or layer name')
    return value


class Document:
    """Only polygon-capable schemas are editable. No implicit schema conversion."""
    def __init__(self, value, *, copy_value=True):
        self.value = copy.deepcopy(value) if copy_value else value
        self._geometries = {}
        if isinstance(value, dict) and value.get('type') == 'FeatureCollection':
            self.kind = 'geojson'
            self.items = self.value.get('features')
        elif isinstance(value, dict) and 'elements' in value.get('annotation', value):
            self.kind = 'histomics'
            self.items = self.value.get('annotation', self.value)['elements']
        else:
            raise ValueError('Supported schemas: GeoJSON FeatureCollection and Histomics annotation.elements/elements')
        if not isinstance(self.items, list):
            raise ValueError('Annotation regions must be a list')
        ids = set()
        for i, item in enumerate(self.items):
            if not isinstance(item, dict):
                raise ValueError('Annotation region must be an object')
            region_id = self.identifier(i)
            if region_id in ids:
                raise ValueError('Duplicate region ID')
            ids.add(region_id)
            self.geometry(i)

    def identifier(self, i):
        item = self.items[i]
        if item.get('id') is not None:
            return str(item['id'])
        return str(item.get('_histoloop_id') or
                   'region-' + hashlib.sha256(json.dumps([i, item], sort_keys=True).encode()).hexdigest()[:20])

    def geometry(self, i):
        if i in self._geometries:
            return self._geometries[i]
        item = self.items[i]
        if self.kind == 'geojson':
            geometry = item.get('geometry', {})
            if item.get('type') != 'Feature' or geometry.get('type') not in ('Polygon', 'MultiPolygon'):
                raise ValueError('Only Polygon/MultiPolygon features are supported; other geometry is not discarded')
            result = shape(geometry)
        else:
            if item.get('type') != 'polyline' or item.get('closed') is not True:
                raise ValueError('Only closed Histomics polylines are supported; convert explicitly before editing')
            points = item.get('points', [])
            if len(points) < 3 or any(not isinstance(p, list) or len(p) not in (2, 3) for p in points):
                raise ValueError('Polyline requires at least three [x,y] or [x,y,z] points')
            result = Polygon([p[:2] for p in points])
        # Invalid input is reported, never repaired behind the reviewer’s back.
        import math
        if not result.is_empty and any(not math.isfinite(v) for v in result.bounds):
            raise ValueError('Geometry contains nonfinite coordinates')
        if not result.is_valid:
            raise ValueError('Invalid geometry: ' + explain_validity(result))
        self._geometries[i] = result
        return result

    def regions(self):
        return [{'id': self.identifier(i), 'name': str(item.get('properties', {}).get('name', item.get('_histoloop_name', item.get('label', '')))),
                 'geometry': mapping(self.geometry(i)), 'bounds': list(self.geometry(i).bounds)}
                for i, item in enumerate(self.items)]

    def union(self):
        return unary_union([self.geometry(i) for i in range(len(self.items))]) if self.items else GeometryCollection()

    def edit(self, operations):
        if not isinstance(operations, list) or len(operations) > 1000:
            raise ValueError('Supply at most 1000 region operations')
        # Assign persistent IDs before changing indices, including untouched regions.
        for i, item in enumerate(self.items):
            if item.get('id') is None and not item.get('_histoloop_id'):
                item['_histoloop_id'] = self.identifier(i)
        for op in operations:
            action = op.get('action')
            index = next((i for i in range(len(self.items)) if self.identifier(i) == op.get('id')), None)
            if action != 'add' and index is None:
                raise Conflict('Region no longer exists')
            if action == 'delete':
                self.items.pop(index)
                continue
            if action not in ('add', 'replace', 'rename'):
                raise ValueError('Unknown region operation')
            item = ({'type': 'Feature', 'properties': {}} if self.kind == 'geojson' else
                    {'type': 'polyline', 'closed': True, 'points': [], 'lineColor': 'rgb(82,156,255)', 'lineWidth': 2}) if action == 'add' else self.items[index]
            if action == 'add':
                item['_histoloop_id'] = uuid.uuid4().hex
            if action != 'rename':
                geometry = op.get('geometry', {})
                if geometry.get('type') not in ('Polygon', 'MultiPolygon'):
                    raise ValueError('Use Polygon or MultiPolygon geometry')
                candidate = shape(geometry)
                if candidate.is_empty or not candidate.is_valid:
                    raise ValueError('Edited geometry is empty or invalid: ' + explain_validity(candidate))
                if self.kind == 'geojson':
                    item['geometry'] = copy.deepcopy(geometry)
                else:
                    if geometry['type'] != 'Polygon' or len(geometry.get('coordinates', [])) != 1:
                        raise ValueError('Histomics polylines cannot represent holes/multipolygons; export and explicitly convert to GeoJSON first')
                    old = item.get('points', [])
                    points = list(candidate.exterior.coords)[:-1]
                    item['points'] = [[float(p[0]), float(p[1]), old[i][2] if i < len(old) and len(old[i]) == 3 else 0]
                                      for i, p in enumerate(points)]
            if 'name' in op:
                if self.kind == 'geojson':
                    item.setdefault('properties', {})['name'] = str(op['name'])[:200]
                else:
                    item['_histoloop_name'] = str(op['name'])[:200]
            if action == 'add':
                self.items.append(item)
        for key in ('_elementCount',):
            if key in self.value:
                self.value[key] = len(self.items)
        Document(self.value)
        return self.value


def feature_collection(geometry, provenance):
    polygons = [] if geometry.is_empty else ([geometry] if geometry.geom_type == 'Polygon' else list(geometry.geoms))
    return {'type': 'FeatureCollection', 'histoloop': provenance,
            'features': [{'type': 'Feature', 'id': hashlib.sha256(p.normalize().wkb).hexdigest()[:24], 'properties': {}, 'geometry': mapping(p)}
                         for p in polygons if p.geom_type == 'Polygon' and not p.is_empty]}


class AnnotationFiles:
    def __init__(self, config, catalog):
        self.root = Path(config.annotation_root or Path(config.state_dir) / 'annotations').resolve()
        self.contours = Path(config.contour_root).resolve() if config.contour_root else None
        self.catalog = catalog
        self._mutex = threading.RLock()

    def contained(self, path):
        # A symlink must never redirect one slide's editable file into another.
        unresolved = Path(path).absolute()
        if not unresolved.is_relative_to(self.root):
            raise ValueError('Annotation path escapes configured root')
        cursor = self.root
        for part in unresolved.relative_to(self.root).parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError('Symlinks inside the annotation root are not editable')
        path = path.resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Annotation path escapes configured root')
        return path

    @contextmanager
    def lock(self, key='registry'):
        self.root.mkdir(parents=True, exist_ok=True)
        with self._mutex:
            path = self.contained(self.root / '.history' / (hashlib.sha256(key.encode()).hexdigest() + '.lock'))
            path.parent.mkdir(exist_ok=True)
            with path.open('a') as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def folder(self, slide_id):
        slide = self.catalog.slide(slide_id)
        identity = str(Path(slide['wsi_path']).resolve())
        with self.lock():
            registry = read_json(self.root / '.samples.json', {})
            entry = registry.get(slide_id)
            if entry and entry['identity'] != identity:
                raise Conflict('Working sample identity changed')
            if not entry:
                name = safe_name(slide['name'])
                duplicate = sum(s['name'] == name for s in self.catalog.slides.values()) > 1
                folder = name + ('--' + hashlib.sha256(identity.encode()).hexdigest()[:12] if duplicate else '')
                owners = [v for v in registry.values() if v['folder'] == folder]
                if owners and any(v['identity'] != identity for v in owners):
                    raise Conflict('Sample folder is owned by another slide')
                directory = self.contained(self.root / folder)
                if directory.exists() and any(directory.glob('*.json')) and not owners:
                    raise Conflict('Unregistered annotation folder requires explicit identity mapping during migration')
                registry[slide_id] = {'folder': folder, 'identity': identity, 'dataset': slide['dataset_id']}
                atomic_json(self.root / '.samples.json', registry)
                entry = registry[slide_id]
            path = self.contained(self.root / entry['folder'])
            path.mkdir(exist_ok=True)
            return path

    def layer_path(self, slide_id, layer):
        safe_name(layer)
        if not layer.endswith('.json') or layer.startswith('.'):
            raise ValueError('Working layers use a visible .json filename')
        return self.contained(self.folder(slide_id) / layer)

    def backup(self, path):
        revision = checksum(path)
        if revision:
            target = self.contained(self.root / '.history' / path.parent.name / path.name / (revision + '.json'))
            if not target.exists():
                atomic_json(target, json.loads(path.read_text()))

    def versions(self, slide_id, layer):
        path = self.layer_path(slide_id, layer)
        directory = self.contained(self.root / '.history' / path.parent.name / path.name)
        return [{'hash': p.stem, 'time_ns': p.stat().st_mtime_ns} for p in sorted(directory.glob('*.json'), key=lambda p:p.stat().st_mtime_ns, reverse=True)[:100]]

    def restore(self, slide_id, layer, payload):
        version = str(payload.get('version', ''))
        if not re.fullmatch('[a-f0-9]{64}', version):
            raise ValueError('Invalid backup revision')
        path = self.layer_path(slide_id, layer)
        backup = self.contained(self.root / '.history' / path.parent.name / path.name / (version + '.json'))
        if not backup.is_file():
            raise FileNotFoundError('Backup does not exist')
        return self.save(slide_id, layer, {**payload, 'document': read_json(backup)})

    def tissue_candidates(self, slide_id):
        folder = self.folder(slide_id)
        paths = [folder / 'Tissue_qc.json'] if (folder / 'Tissue_qc.json').is_file() else []
        slide = self.catalog.slide(slide_id)
        if self.contours and self.contours.is_dir():
            for suffix in ('.json', '.geojson'):
                paths.extend(self.contours.glob(slide['name'] + suffix))
                paths.extend(self.contours.glob('*/' + slide['name'] + suffix))
        return {hashlib.sha256(str(p.resolve()).encode()).hexdigest()[:20]: p for p in paths}

    def tissue(self, slide_id, selection=None):
        candidates = self.tissue_candidates(slide_id)
        saved = read_json(self.folder(slide_id) / '.tissue.json', {}).get('id')
        choice = selection or saved
        if not choice and len(candidates) == 1 and sum(s['name'] == self.catalog.slide(slide_id)['name'] for s in self.catalog.slides.values()) == 1:
            choice = next(iter(candidates))
        if choice not in candidates:
            raise ValueError('Select a tissue mask explicitly' if candidates else 'Tissue mask is missing')
        path = self.tissue_path(candidates[choice])
        return choice, path, Document(read_json(path))

    def tissue_path(self, path):
        if Path(path).absolute().is_relative_to(self.root):
            return self.contained(path)
        resolved = Path(path).resolve()
        if not self.contours or not resolved.is_relative_to(self.contours):
            raise ValueError('Tissue path escapes configured contour root')
        return resolved

    def copy_tissue_to_working(self, slide_id, selection, reviewer=None):
        """Copy a selected configured contour into the editable per-slide tissue file."""
        candidates = self.tissue_candidates(slide_id)
        source = candidates.get(str(selection or ''))
        if source is None:
            raise ValueError('Select an available tissue mask')
        source = self.tissue_path(source)
        folder = self.folder(slide_id)
        target = self.layer_path(slide_id, 'Tissue_qc.json')
        if source == target:
            raise ValueError('Tissue_qc.json is already the editable working layer')
        result = self.save(slide_id, 'Tissue_qc.json', {
            'revision': None,
            'document': Document(read_json(source)).value,
            'reviewer': reviewer,
        })
        working_id = next((key for key, path in self.tissue_candidates(slide_id).items()
                           if self.tissue_path(path) == target), None)
        if working_id is None:
            raise RuntimeError('The editable tissue copy was saved but could not be selected')
        atomic_json(folder / '.tissue.json', {
            'id': working_id,
            'copied_from': source.name,
            'source_revision': checksum(source),
        })
        return {**result, 'tissue_id': working_id, 'source_name': source.name}

    def snapshot(self, slide_id, documents=None):
        folder = self.folder(slide_id)
        hashes = {}
        for name in SOURCE_NAMES[:2]:
            path = self.contained(folder / name)
            raw = path.read_bytes() if path.is_file() else None
            hashes[name] = hashlib.sha256(raw).hexdigest() if raw is not None else None
            if documents is not None:
                documents[name] = Document(json.loads(raw)) if raw is not None else None
        choice, path, _ = self.tissue(slide_id)
        raw = path.read_bytes()
        return {**hashes, 'tissue_id': choice, 'tissue_hash': hashlib.sha256(raw).hexdigest()}, Document(json.loads(raw))

    def list(self, slide_id):
        folder = self.folder(slide_id)
        generation = read_json(folder / '.generation.json', {'status': 'unavailable'})
        try:
            snapshot, _ = self.snapshot(slide_id)
            if generation.get('sources') != snapshot:
                generation = {**generation, 'status': 'stale'}
        except ValueError as error:
            generation = {**generation, 'status': 'unavailable', 'error': str(error)}
        layers = []
        paths = {p.name: p for p in folder.glob('*.json') if not p.name.startswith('.')}
        for name in (*SOURCE_NAMES[:2], *DERIVED_NAMES):
            paths.setdefault(name, folder / name)
        for name, path in paths.items():
            item = {'id': name, 'name': name, 'editable': name not in DERIVED_NAMES,
                    'derived': name in DERIVED_NAMES, 'revision': None, 'present': path.is_file(), 'count': 0}
            try:
                path = self.contained(path)
                item['revision'] = checksum(path)
                if path.is_file():
                    item['count'] = len(Document(read_json(path)).items)
                item['status'] = generation['status'] if item['derived'] else 'saved' if item['present'] else 'missing'
            except (ValueError, TypeError, KeyError) as error:
                item.update(status='error', error=str(error))
            layers.append(item)
        tissues = []
        for key, path in self.tissue_candidates(slide_id).items():
            tissues.append({'id': key, 'name': path.name})
            if path.parent != folder:
                try:
                    path = self.tissue_path(path)
                    layers.append({'id': 'tissue:' + key, 'name': path.name, 'editable': False, 'derived': False,
                                   'revision': checksum(path), 'present': True, 'count': len(Document(read_json(path)).items), 'status': 'saved'})
                except ValueError as error:
                    layers.append({'id': 'tissue:' + key, 'name': path.name, 'editable': False, 'present': True, 'count': 0, 'status': 'error', 'error': str(error)})
        return {'layers': layers, 'tissues': tissues, 'generation': generation,
                'selected_tissue': read_json(folder / '.tissue.json', {}).get('id')}

    def read(self, slide_id, layer):
        if layer.startswith('tissue:'):
            path = self.tissue_candidates(slide_id).get(layer[7:])
            if not path:
                raise KeyError('Unknown tissue identity')
            path = self.tissue_path(path)
        else:
            path = self.layer_path(slide_id, layer)
        if not path.is_file():
            raise FileNotFoundError('Annotation file is missing')
        raw = path.read_bytes()
        doc = Document(json.loads(raw))
        return {'revision': hashlib.sha256(raw).hexdigest(), 'schema': doc.kind, 'regions': doc.regions(), 'document': doc.value}

    def save(self, slide_id, layer, payload):
        if layer in DERIVED_NAMES:
            raise ValueError('Derived layers are read-only; edit ground-truth files')
        path = self.layer_path(slide_id, layer)
        with self.lock(path.parent.name):
            previous = checksum(path)
            if 'revision' not in payload or payload['revision'] != previous:
                raise Conflict('Annotation changed externally or in another window; draft retained. Reload and reconcile before retrying.')
            if 'document' in payload:
                document = Document(payload['document']).value
            elif payload.get('create') and previous is None:
                document = {'type': 'FeatureCollection', 'features': []}
            elif payload.get('remove'):
                self.backup(path)
                path.unlink(missing_ok=True)
                atomic_json(path.parent / '.generation.json', {'status': 'stale', 'reason': 'Source removed', 'time': now()})
                return {'revision': None, 'status': 'missing'}
            else:
                if previous is None:
                    raise ValueError('Explicitly create or import a missing file first')
                document = Document(read_json(path)).edit(payload.get('operations', []))
            self.backup(path)
            if checksum(path) != previous:
                raise Conflict('Annotation changed externally while saving; draft retained')
            atomic_json(path, document)
            atomic_json(self.root / '.history' / path.parent.name / 'latest-save.json',
                        {'layer': layer, 'previous': previous, 'revision': checksum(path), 'reviewer': payload.get('reviewer'), 'time': now()})
            if layer in SOURCE_NAMES:
                atomic_json(path.parent / '.generation.json', {'status': 'stale', 'time': now()})
            doc = Document(document)
            return {'revision': checksum(path), 'status': 'saved', 'document': document, 'schema': doc.kind, 'regions': doc.regions()}

    def regenerate(self, slide_id, selection=None):
        folder = self.folder(slide_id)
        with self.lock(folder.name):
            if selection:
                self.tissue(slide_id, selection)
                atomic_json(folder / '.tissue.json', {'id': selection})
            try:
                documents = {}
                sources, tissue = self.snapshot(slide_id, documents)
                c = documents[SOURCE_NAMES[0]].union() if documents[SOURCE_NAMES[0]] else GeometryCollection()
                m = documents[SOURCE_NAMES[1]].union() if documents[SOURCE_NAMES[1]] else GeometryCollection()
                t = tissue.union()
                geometries = {'cortical.json': c.difference(m).intersection(t), 'medullary.json': m.intersection(t)}
            except (ValueError, TypeError, KeyError) as error:
                manifest = {'status': 'unavailable', 'error': str(error), 'time': now()}
                atomic_json(folder / '.generation.json', manifest)
                return manifest
            generation_id = uuid.uuid4().hex
            stage = self.contained(self.root / '.history' / folder.name / 'generations' / generation_id)
            manifest = {'id': generation_id, 'status': 'pending', 'sources': sources, 'algorithm': ALGORITHM, 'time': now(), 'outputs': {}}
            for name, geometry in geometries.items():
                source = SOURCE_NAMES[0] if name == 'cortical.json' else SOURCE_NAMES[1]
                present = bool(sources[source])
                manifest['outputs'][name] = present
                if present:
                    provenance = {k: v for k, v in manifest.items() if k not in ('status', 'outputs')}
                    atomic_json(stage / name, feature_collection(geometry, provenance))
            atomic_json(folder / '.generation.json', manifest)
            return self._publish(slide_id, manifest)

    def _publish(self, slide_id, manifest):
        folder = self.folder(slide_id)
        latest, _ = self.snapshot(slide_id)
        if latest != manifest['sources']:
            manifest['status'] = 'stale'
            atomic_json(folder / '.generation.json', manifest)
            return manifest
        stage = self.contained(self.root / '.history' / folder.name / 'generations' / manifest['id'])
        for name, present in manifest['outputs'].items():
            self.backup(folder / name)
            if present:
                atomic_json(folder / name, read_json(stage / name))
            else:
                (folder / name).unlink(missing_ok=True)
        manifest['status'] = 'current'
        atomic_json(folder / '.generation.json', manifest)
        return manifest

    def recover(self, slide_id):
        folder = self.folder(slide_id)
        with self.lock(folder.name):
            manifest = read_json(folder / '.generation.json', {})
            if manifest.get('status') == 'pending':
                try:
                    return self._publish(slide_id, manifest)
                except (ValueError, FileNotFoundError):
                    manifest['status'] = 'stale'
                    atomic_json(folder / '.generation.json', manifest)
        if manifest.get('status') == 'stale':
            return self.regenerate(slide_id)

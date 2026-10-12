"""Dry-run-first, identity-aware annotation migration. Reports contain private paths."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from .annotation_files import AnnotationFiles, Document, SOURCE_NAMES, checksum, safe_name
from .config import Config
from .catalog import Catalog
from .storage import Store, atomic_json, now, read_json

NATIVE_SOURCES = [f'/blue/pinaki.sarder/j.fermin/Cort_Med_Project/KPMP_{c}/Labelled_files' for c in ('barisoni', 'pathologist3', 'hodgin')] + [f'/orange/pinaki.sarder/j.fermin/cort_med_wsi/KPMP_batch{n}/Labelled_files' for n in range(2, 7)]


def plan(config, catalog, progress=None):
    roots = json.loads(config.annotation_sources or '[]')
    root = Path(config.annotation_root or Path(config.state_dir) / 'annotations').resolve()
    samples, candidates = {}, defaultdict(lambda: defaultdict(list))
    report = {'version': 1, 'time': now(), 'destination': str(root), 'missing_roots': [], 'malformed': [],
              'unmatched': [], 'ambiguous': [], 'duplicate_names': [], 'conflicts': [], 'samples': [], 'copies': [], 'preserved': []}
    registry = read_json(root / '.samples.json', {})
    slides = list(catalog.slides.values())
    names = Counter(s['name'] for s in slides)
    for s in slides:
        identity = str(Path(s['wsi_path']).resolve())
        samples[s['id']] = {'name': s['name'], 'identity': identity, 'dataset': s['dataset_id'], 'kind': 'manifest'}
    for source in roots:
        source = Path(source).resolve()
        if not source.is_dir():
            report['missing_roots'].append(str(source))
            continue
        for folder in sorted(p for p in source.iterdir() if p.is_dir()):
            name = folder.name[:-4] if folder.name.lower().endswith('.svs') else folder.name
            try:
                safe_name(name)
            except ValueError as error:
                report['ambiguous'].append({'source': str(folder), 'reason': str(error)})
                continue
            matches = [s for s in slides if s['name'] == name and source.parent.name in Path(s['wsi_path']).parts]
            if not matches:
                by_name = [s for s in slides if s['name'] == name]
                # A source cohort must match a slide cohort; basename alone is insufficient.
                if by_name:
                    report['ambiguous'].append({'source': str(folder), 'slides': [s['id'] for s in by_name], 'reason': 'Cohort identity does not match'})
                report['unmatched'].append(str(folder))
                key = 'source-' + hashlib.sha256(str(folder).encode()).hexdigest()[:20]
                samples[key] = {'name': name, 'identity': str(folder), 'dataset': source.parent.name, 'kind': 'source-only'}
            elif len(matches) > 1:
                report['ambiguous'].append({'source': str(folder), 'slides': [s['id'] for s in matches]})
                key = 'source-' + hashlib.sha256(str(folder).encode()).hexdigest()[:20]
                samples[key] = {'name': name, 'identity': str(folder), 'dataset': source.parent.name, 'kind': 'source-only'}
            else:
                key = matches[0]['id']
            alias = folder / 'cortex_gt.json'
            for layer in SOURCE_NAMES[:2]:
                files = [folder / layer]
                if layer == 'cortical_gt.json':
                    files.append(alias)
                for file in files:
                    if not file.is_file():
                        continue
                    try:
                        raw = file.read_bytes()
                        Document(json.loads(raw), copy_value=False)
                        candidates[key][layer].append({'source': str(file), 'sha256': hashlib.sha256(raw).hexdigest()})
                    except (ValueError, KeyError, TypeError) as error:
                        report['malformed'].append({'source': str(file), 'reason': str(error)})
                if layer == 'cortical_gt.json' and alias.exists() and (folder / layer).exists():
                    report['conflicts'].append({'sample': key, 'layer': layer, 'reason': 'Both cortical_gt.json and cortex_gt.json exist'})
        if progress:
            progress({'root': str(source), 'sample_identities': len(samples), 'malformed': len(report['malformed'])})
    counts = Counter(s['name'] for s in samples.values())
    report['duplicate_names'] = sorted(n for n, count in counts.items() if count > 1)
    for key, sample in samples.items():
        existing = registry.get(key)
        directory = sample['name'] + ('--' + hashlib.sha256(sample['identity'].encode()).hexdigest()[:12] if counts[sample['name']] > 1 else '')
        if existing:
            if existing['identity'] != sample['identity']:
                report['conflicts'].append({'sample': key, 'reason': 'Registered identity changed'})
                continue
            directory = existing['folder']
        target = root / directory
        if not target.resolve().is_relative_to(root):
            report['conflicts'].append({'sample': key, 'reason': 'Destination symlink escapes root'})
            continue
        if not existing and target.exists() and any(target.glob('*.json')):
            report['conflicts'].append({'sample': key, 'reason': 'Unregistered destination has edits; identity must be resolved explicitly'})
            continue
        report['samples'].append({'id': key, **sample, 'folder': directory})
        for layer, files in candidates[key].items():
            if any(c.get('sample') == key and c.get('layer') == layer for c in report['conflicts']):
                continue
            if len(files) > 1:
                report['conflicts'].append({'sample': key, 'layer': layer, 'sources': files})
                continue
            entry = {**files[0], 'sample': key, 'layer': layer, 'destination': str(target / layer)}
            if (target / layer).exists():
                report['preserved'].append({**entry, 'destination_sha256': checksum(target / layer)})
            else:
                report['copies'].append(entry)
    return report


def execute(report):
    """Apply a reviewed plan; changed sources cause an error, existing files are never replaced."""
    root = Path(report['destination']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    (root / '.history').mkdir(exist_ok=True)
    with (root / '.history' / (hashlib.sha256(b'registry').hexdigest() + '.lock')).open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        registry = read_json(root / '.samples.json', {})
        for s in report['samples']:
            if s['id'] in registry and registry[s['id']]['identity'] != s['identity']:
                raise ValueError('Migration identity changed since dry run')
            folder = root / safe_name(s['folder'])
            if not folder.resolve().is_relative_to(root):
                raise ValueError('Destination escaped root')
            folder.mkdir(exist_ok=True)
            registry[s['id']] = {k: s[k] for k in ('folder', 'identity', 'dataset', 'kind')}
        atomic_json(root / '.samples.json', registry)
        copied = []
        for entry in report['copies']:
            source, target = Path(entry['source']), Path(entry['destination'])
            if not target.resolve().is_relative_to(root) or target.name not in SOURCE_NAMES:
                raise ValueError('Invalid destination in plan')
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise ValueError('Source changed after dry run: ' + str(source))
            if target.exists():
                continue
            # Publish bytes unchanged through an exclusive hard-link; rescan cannot clobber edits.
            temporary = target.parent / ('.migration-' + os.urandom(8).hex())
            try:
                with temporary.open('xb') as handle:
                    handle.write(raw); handle.flush(); os.fsync(handle.fileno())
                try:
                    os.link(temporary, target)
                    descriptor = os.open(target.parent, os.O_RDONLY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                    copied.append(entry)
                except FileExistsError:
                    pass
            finally:
                temporary.unlink(missing_ok=True)
        atomic_json(root / '.history' / 'migration' / (now().replace(':', '-') + '.json'), {'plan': report, 'copied': copied})
        return {'folders': len(report['samples']), 'copied': len(copied), 'conflicts': len(report['conflicts']), 'malformed': len(report['malformed'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Execute the previously written dry-run report')
    parser.add_argument('--native-sources', action='store_true')
    args = parser.parse_args()
    if args.execute:
        print(json.dumps(execute(read_json(args.report))))
        return
    config = Config.from_env()
    if args.native_sources:
        config.annotation_sources = json.dumps(NATIVE_SOURCES)
    catalog = Catalog(config, Store(config.state_dir))
    report = plan(config, catalog, progress=lambda value: print(json.dumps(value), flush=True))
    atomic_json(args.report, report)
    print(json.dumps({key: len(report[key]) for key in ('samples', 'copies', 'preserved', 'missing_roots', 'conflicts', 'ambiguous', 'malformed', 'unmatched')}))


if __name__ == '__main__':
    main()

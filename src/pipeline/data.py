"""Non-sharded input preparation and lazy paired windows; no torch import."""
from pathlib import Path
import hashlib
import json
import os
import platform
import sys
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ('train', 'val', 'actor2_test', 'actor1_test')


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def config_from(path):
    return yaml.safe_load(resolve(path).read_text(encoding='utf-8'))


def semantic_devices(path):
    spec = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if spec['dataset']['name'] != 'CU':
        raise ValueError('This restart preserves the supplied CU semantic model only.')
    indexed = {}
    for state in spec['behavioral_states']:
        for device in state['compatible_devices']:
            index, name = device['index'], device['id']
            if index in indexed and indexed[index] != name:
                raise ValueError('Conflicting semantic device identities.')
            indexed[index] = name
    if sorted(indexed) != list(range(spec['dataset']['selected_device_count'])):
        raise ValueError('Semantic device indices must cover the whole input schema.')
    return [indexed[i] for i in sorted(indexed)]


def prepare(config, *, allow_incomplete=False):
    options = config['data']
    source, output = resolve(options['csv']), resolve(options['output'])
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new data.output directory.')
    semantic = resolve(options['semantic_config'])
    devices = semantic_devices(semantic)
    frame = pd.read_csv(source)
    frame.columns = frame.columns.str.strip()
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate columns after whitespace normalization.')
    missing = set(['Timestamp', *devices]) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    frame = frame[['Timestamp', *devices]].copy()
    frame['Timestamp'] = pd.to_datetime(frame['Timestamp'], errors='raise')
    frame = frame.sort_values('Timestamp').reset_index(drop=True)
    if frame.Timestamp.isna().any() or frame.Timestamp.duplicated().any():
        raise ValueError('Missing or duplicate timestamps; no automatic aggregation applied.')
    # The current protocol is 1-second data. New resampling policies are not invented.
    if options['frequency'] != '1s':
        raise ValueError('This restart requires existing 1s observations; resampling needs an explicit policy.')
    periods = options['periods']
    ranges = sorted((pd.Timestamp(a), pd.Timestamp(b), k) for k, (a, b) in periods.items())
    if any(a > b for a, b, _ in ranges) or any(ranges[i][1] >= ranges[i+1][0] for i in range(len(ranges)-1)):
        raise ValueError('Actor periods must be ordered internally and non-overlapping.')
    subsets = {k: frame[frame.Timestamp.between(a, b)].copy() for a,b,k in ranges}
    ratio = float(options['val_ratio'])
    w = int(options['window_size'])
    if not 0 < ratio < 1 or w < 1:
        raise ValueError('Require 0 < val_ratio < 1 and positive window_size.')
    first = subsets.pop('actor1_t1')
    cut = int(len(first) * (1-ratio))
    subsets.update(train=first.iloc[:cut].copy(), val=first.iloc[cut:].copy())
    if len(subsets['train']) <= w:
        raise ValueError('Training data must contain more rows than window_size.')
    counts = {k: max(0, len(subsets[k])-w) for k in SPLITS}
    short = [k for k,n in counts.items() if n == 0]
    if short and not allow_incomplete:
        raise ValueError(f'No paired windows in {short}. Supply full data; --allow-incomplete is for module checks only.')
    for name, subset in subsets.items():
        ticks = subset.Timestamp.to_numpy(dtype='datetime64[ns]')
        if len(ticks)>1 and np.any(np.diff(ticks) != np.timedelta64(1, 's')):
            raise ValueError(f'{name}: time gaps; windows must not cross missing observations.')
        values = subset[devices].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f'{name}: nonnumeric/missing/nonfinite observations; no implicit imputation.')
    scaler = MinMaxScaler().fit(subsets['train'][devices].to_numpy(dtype=np.float64))
    output.mkdir(parents=True)
    manifest = {'schema_version': 1, 'dataset': 'CU', 'window_size': w,
                'window_target': 'X[i:i+W], Y[i+1:i+W+1]',
                'incomplete': bool(short), 'devices': devices,
                'source_sha256': digest(source), 'semantic_sha256': digest(semantic),
                'config': config, 'python': sys.version, 'platform': platform.platform(),
                'command': sys.argv, 'splits': {}}
    for name in SPLITS:
        subset = subsets[name]
        a = scaler.transform(subset[devices].to_numpy(dtype=np.float64)).astype(np.float32) if len(subset) else np.empty((0,len(devices)), dtype=np.float32)
        p = output / f'{name}.npy'
        np.save(p, a)
        tp = output / f'{name}_timestamps.npy'
        np.save(tp, subset.Timestamp.to_numpy(dtype='datetime64[ns]'))
        manifest['splits'][name] = {'rows':len(a), 'pairs':counts[name],
                                   'sha256':digest(p), 'timestamps_sha256':digest(tp)}
    write_json(output/'devices.json', devices)
    write_json(output/'scaler.json', {k:getattr(scaler,k).tolist() for k in ['data_min_','data_max_','scale_','min_']})
    manifest['devices_sha256'] = digest(output/'devices.json')
    manifest['scaler_sha256'] = digest(output/'scaler.json')
    write_json(output/'dataset.json', manifest)
    return manifest


def load_dataset(config):
    root = resolve(config['data']['output'])
    manifest = json.loads((root/'dataset.json').read_text())
    if manifest['config']['data'] != config['data']:
        raise ValueError('Data configuration changed after preprocessing; use the recorded configuration.')
    if manifest['window_size'] != config['data']['window_size']:
        raise ValueError('Window-size mismatch; regenerate data into a new directory.')
    if digest(resolve(config['data']['semantic_config'])) != manifest['semantic_sha256']:
        raise ValueError('Semantic configuration changed after preprocessing.')
    for filename, field in [('devices.json','devices_sha256'),('scaler.json','scaler_sha256')]:
        if digest(root/filename) != manifest[field]:
            raise ValueError(f'{filename} changed after preprocessing.')
    return root, manifest


def split_array(root, manifest, split):
    path = root/f'{split}.npy'
    if digest(path) != manifest['splits'][split]['sha256']:
        raise ValueError(f'{split} array differs from preprocessing manifest.')
    if digest(root/f'{split}_timestamps.npy') != manifest['splits'][split]['timestamps_sha256']:
        raise ValueError(f'{split} timestamps differ from preprocessing manifest.')
    return np.load(path, mmap_mode='r', allow_pickle=False)


def window_batches(array, w, batch_size, *, paired=True):
    """Batch chronological windows without storing overlapping windows on disk."""
    if w < 1 or batch_size < 1:
        raise ValueError('Positive window and batch sizes required.')
    count = max(0, len(array)-w+(0 if paired else 1))
    for start in range(0, count, batch_size):
        stop = min(count, start+batch_size)
        x = np.stack([array[i:i+w] for i in range(start,stop)])
        y = np.stack([array[i+1:i+w+1] for i in range(start,stop)]) if paired else None
        yield start, x, y

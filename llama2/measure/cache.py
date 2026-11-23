"""Content-addressed manifests, atomic artifacts and inter-process exclusion."""
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]


def key(data):
    data = json.dumps(data, sort_keys=True).encode()
    return hashlib.sha256(data).hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def versions():
    return {name: importlib.metadata.version(name) for name in
            ('torch', 'transformers', 'accelerate', 'safetensors', 'numpy')}


def commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


@contextlib.contextmanager
def temporary(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    os.close(fd)
    try:
        yield Path(name)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def json_save(path, data):
    with temporary(path) as tmp:
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def save(path, data):
    with temporary(path) as tmp:
        torch.save(data, tmp)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def checkpoint(path, memo):
    """Hash each immutable input once, refresh after replacement or modification."""
    path = Path(path).resolve(strict=True)
    memo.mkdir(parents=True, exist_ok=True)
    record = memo / (key(str(path)) + '.json')
    files = sorted(p for p in path.iterdir() if p.is_file() and
                   p.suffix in {'.json', '.bin', '.safetensors', '.model', '.yaml'})
    if not any(p.suffix in {'.bin', '.safetensors'} for p in files):
        raise ValueError(f'No model weights in {path}')
    with lock(record.with_suffix('.lock')):
        old = json.loads(record.read_text()) if record.exists() else {}
        entries = {}
        for p in files:
            stat = p.stat()
            stamp = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
            item = old.get(p.name, {})
            entries[p.name] = item if item.get('stat') == stamp else {
                'stat': stamp, 'sha256': digest(p)}
        if entries != old:
            json_save(record, entries)
    hashes = {name: value['sha256'] for name, value in entries.items()}
    return {'path': str(path), 'revision': key(hashes), 'files': hashes}


def register(run, base_path, ft_path, data, params, train_commit=None, train_versions=None):
    run = Path(run)
    folder = run / 'sft'
    with lock(folder / 'register.lock'):
        base = checkpoint(base_path, run / '.hashes')
        ft = checkpoint(ft_path, run / '.hashes')
        manifest = {'base': base, 'ft': ft, 'training_data': str(Path(data).resolve()),
                    'training_data_sha256': digest(data), 'registered_at_commit': commit(),
                    'registration_versions': versions(), 'training_commit': train_commit,
                    'train_params': Path(params).read_text() if params else None,
                    'training_versions': Path(train_versions).read_text() if train_versions else None,
                    'format': 'adapter' if 'adapter_config.json' in ft['files'] else 'full'}
        target = folder / 'manifest.json'
        if target.exists():
            old = json.loads(target.read_text())
            for field in ('base', 'ft', 'training_data_sha256', 'train_params', 'training_commit', 'training_versions'):
                if old[field] != manifest[field]:
                    raise ValueError(f'Existing run has different {field}; use another run directory')
        else:
            json_save(target, manifest)
        weights = folder / 'checkpoint'
        if not weights.is_symlink():
            weights.symlink_to(ft['path'], target_is_directory=True)
        if weights.resolve(strict=True) != Path(ft['path']):
            raise ValueError(f'Unexpected checkpoint link: {weights}')
    print(folder)

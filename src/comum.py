from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import duckdb


def agora():
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def assinatura(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def ler_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def escrever_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def q(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def ident(name):
    return '"' + name.replace('"', '""') + '"'


def parquet(files):
    files = list(files)
    if not files:
        raise ValueError('Lista de Parquets vazia.')
    return 'read_parquet([' + ','.join(q(Path(p).as_posix()) for p in files) + '], hive_partitioning=false, union_by_name=false)'


def conectar(temp_dir, memory_limit='32GB', threads=8):
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(':memory:')
    con.execute(f'SET threads={int(threads)}')
    con.execute(f'SET memory_limit={q(memory_limit)}')
    con.execute(f'SET temp_directory={q(temp_dir.as_posix())}')
    con.execute("SET TimeZone='UTC'")
    con.execute('SET preserve_insertion_order=false')
    return con


def exportar(con, query, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + '.partial')
    if temp.exists():
        temp.unlink()
    con.execute(f'COPY ({query}) TO {q(temp.as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)')
    temp.replace(target)


def dividir(a, b):
    return a / b if b else None


def ambiente(root):
    import importlib.metadata
    import psutil
    packages = ['duckdb', 'numpy', 'pyarrow', 'torch', 'scikit-learn', 'joblib', 'matplotlib', 'psutil']
    versions = {name: importlib.metadata.version(name) for name in packages}
    def git(*args):
        return subprocess.check_output(['git', '--no-optional-locks', '-C', str(root), *args], text=True).strip()
    try:
        commit, dirty = git('rev-parse', 'HEAD'), git('status', '--porcelain')
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    try:
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        gpu = None
    return dict(python=platform.python_version(), packages=versions, os=platform.platform(), cpu=platform.processor(), logical_cpus=os.cpu_count(), ram_bytes=psutil.virtual_memory().total, gpu=gpu, git_commit=commit, git_status=dirty)


def mcc(tp, fp, fn, tn):
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return (tp * tn - fp * fn) / math.sqrt(denominator) if denominator else None

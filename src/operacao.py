from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psutil
import pyarrow.parquet as pq

from .comum import agora, escrever_json, ler_json


def conferir_arquivos(files, expected_files, expected_rows):
    rows = 0
    schemas = set()
    for path in files:
        source = pq.ParquetFile(path)
        rows += source.metadata.num_rows
        schemas.add(str(source.schema_arrow))
    if len(files) != expected_files or rows != expected_rows or len(schemas) != 1:
        raise ValueError(f'Inventário diverge do aprovado: {len(files)} arquivos, {rows} registros, {len(schemas)} esquemas.')
    return dict(files=len(files),rows=rows,bytes=sum(p.stat().st_size for p in files),schema_count=len(schemas))


def processo_ativo(lock):
    try:
        record = ler_json(lock)
        return psutil.Process(record['pid']).create_time() == record['created']
    except (FileNotFoundError,psutil.NoSuchProcess):
        return False
    except psutil.AccessDenied:
        return True


def manter_acordado(enabled):
    if enabled and sys.platform == 'win32':
        if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
            raise OSError('Não foi possível inibir a suspensão automática.')
        return True
    return False


def liberar_suspensao(active):
    if active:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def encerrar_familia(child):
    if child is not None and child.poll() is None:
        try:
            parent=psutil.Process(child.pid)
            for process in reversed(parent.children(recursive=True)):
                try:process.terminate()
                except psutil.NoSuchProcess:pass
            parent.terminate()
            child.wait(timeout=10)
        except psutil.NoSuchProcess:
            pass


def monitorar(command, root, run_path, operations, interval=5, prevent_sleep=True):
    """Supervisiona uma tentativa; não faz repetição automática após falha."""
    root,run_path,operations = map(Path,(root,run_path,operations))
    operations.mkdir(parents=True,exist_ok=True)
    lock = operations/'.supervisor.lock'
    if lock.exists():
        if processo_ativo(lock):
            raise RuntimeError('Já existe um supervisor ativo para esta execução.')
        lock.unlink()
    descriptor = os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    with os.fdopen(descriptor,'w',encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(),created=psutil.Process().create_time()),stream)
        stream.flush();os.fsync(stream.fileno())
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]
    log_path,telemetry = operations/f'{stamp}.log',operations/f'{stamp}-recursos.jsonl'
    attempt_path = operations/f'{stamp}-estado.json'
    state = dict(attempt=stamp,status='starting',started_at=agora(),command=command,run=str(run_path),log=str(log_path),resources=str(telemetry),supervisor_pid=os.getpid())
    def save_state():
        escrever_json(attempt_path,state)
        escrever_json(operations/'estado.json',state)
    stop = threading.Event();sampler=None;child=None;awake=False
    start = time.perf_counter()
    try:
        with log_path.open('a',encoding='utf-8',buffering=1) as log:
            def message(line):
                log.write(line+'\n');log.flush()
                print(line,flush=True)
            save_state()
            if prevent_sleep:
                try:
                    awake=manter_acordado(True)
                except OSError as exc:
                    message('Aviso: '+str(exc))
            message(f'Execução: {run_path.name}\nLog: {log_path}')
            env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONUNBUFFERED='1')
            child=subprocess.Popen(command,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env=env,bufsize=1)
            state.update(status='running',child_pid=child.pid,sleep_inhibited=awake)
            save_state()
            def sample():
                try:
                    with telemetry.open('a',encoding='utf-8',buffering=1) as stream:
                        while not stop.is_set():
                            entry=dict(at_utc=agora(),elapsed_seconds=time.perf_counter()-start,disk_free_bytes=psutil.disk_usage(str(root)).free,
                                       system_available_memory=psutil.virtual_memory().available)
                            try:
                                pipeline_lock=run_path/'.lock'
                                pid=ler_json(pipeline_lock)['pid'] if pipeline_lock.exists() else child.pid
                                process=psutil.Process(pid)
                                entry.update(pid=pid,memory=process.memory_info()._asdict(),cpu_times=process.cpu_times()._asdict(),io=process.io_counters()._asdict())
                                result=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4)
                                if result.returncode==0:
                                    used,util=result.stdout.strip().split(',')
                                    entry['gpu']=dict(board_used_mib=float(used),utilization_percent=float(util))
                            except (psutil.Error,OSError,ValueError,subprocess.SubprocessError,json.JSONDecodeError):
                                pass
                            stream.write(json.dumps(entry)+'\n');stream.flush()
                            escrever_json(operations/'monitor.json',entry)
                            stop.wait(interval)
                except Exception as exc:
                    # A falha do monitor não encerra o cálculo ou oculta seu código de saída.
                    with (operations/f'{stamp}-monitor-error.log').open('a',encoding='utf-8') as errors:
                        errors.write(str(exc)+'\n')
            sampler=threading.Thread(target=sample,daemon=True);sampler.start()
            try:
                for line in child.stdout:
                    message(line.rstrip('\r\n'))
                code=child.wait()
            except KeyboardInterrupt:
                message('Interrupção solicitada. Preservando os checkpoints já gravados.')
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    encerrar_familia(child)
                code=130
            status='completed' if code==0 else ('interrupted' if code==130 else 'failed')
            if code==0:
                manifest=ler_json(run_path/'manifesto.json')
                if manifest['status']!='completed':
                    raise RuntimeError('Processo encerrou sem concluir o manifesto do pipeline.')
            state.update(status=status,exit_code=code,ended_at=agora(),elapsed_seconds=time.perf_counter()-start)
            save_state()
            message('Concluído. Retorne para a auditoria dos resultados.' if code==0 else f'Processamento encerrado com código {code}. Retorne com o log para a contingência.')
            return code
    except BaseException as exc:
        state.update(status='failed',ended_at=agora(),error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.perf_counter()-start)
        save_state()
        raise
    finally:
        stop.set()
        if sampler is not None:sampler.join(timeout=6)
        encerrar_familia(child)
        liberar_suspensao(awake)
        lock.unlink(missing_ok=True)

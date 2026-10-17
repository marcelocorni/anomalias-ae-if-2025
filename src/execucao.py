from __future__ import annotations

import os
import re
import time
from pathlib import Path

import psutil

from .comum import agora, ambiente, assinatura, escrever_json, ler_json, sha256


def codigo(root):
    root=Path(root)
    files=sorted(root.glob('src/*.py'))+sorted(root.glob('scripts/*.py'))+[root/'pyproject.toml',root/'uv.lock']
    return {str(p.relative_to(root)):sha256(p) for p in files}


class Execucao:
    def __init__(self, root, output, identifier, config, paths, selected_days, mode):
        if not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*', identifier):
            raise ValueError('Identificador da execução inválido.')
        self.root = Path(root).resolve()
        self.path = (Path(output)/identifier).resolve()
        if not self.path.is_relative_to(Path(output).resolve()):
            raise ValueError('Execução fora do diretório de resultados.')
        self.cache = {}
        self.integrity = config.get('integridade', {})
        code = codigo(self.root)
        self.environment = ambiente(root)
        identity = dict(config=config,paths={k:str(v) for k,v in paths.items()},days=selected_days,mode=mode,
                        code=code,packages=self.environment['packages'],python=self.environment['python'])
        self.signature = assinatura(identity)
        self.path.mkdir(parents=True, exist_ok=True)
        self.lock = self.path/'.lock'
        if self.lock.exists():
            lock = ler_json(self.lock)
            try:
                active = psutil.Process(lock['pid']).create_time() == lock['created']
            except psutil.NoSuchProcess:
                active = False
            if active:
                raise RuntimeError('Execução já está ativa em outro processo.')
            self.lock.unlink()
        descriptor = os.open(self.lock, os.O_WRONLY|os.O_CREAT|os.O_EXCL)
        with os.fdopen(descriptor, 'w') as stream:
            import json
            json.dump(dict(pid=os.getpid(),created=psutil.Process().create_time()),stream)
        try:
            manifest = self.path/'manifesto.json'
            if manifest.exists():
                self.manifest = ler_json(manifest)
                if self.manifest['signature'] != self.signature:
                    continuation=self.path/'continuidade.json'
                    accepted=ler_json(continuation) if continuation.exists() else {}
                    if (self.manifest.get('scientific_status')!='completed'
                        or accepted.get('source_signature')!=self.manifest['signature']
                        or accepted.get('identity')!=identity
                        or accepted.get('signature')!=assinatura(identity)):
                        raise ValueError('Configuração, código ou ambiente mudou; escolha outro identificador de execução.')
                    self.signature=self.manifest['signature']
            else:
                if any(p.name not in ('.lock','operacao') for p in self.path.iterdir()):
                    raise ValueError('Diretório existente sem manifesto; não será reutilizado.')
                self.manifest = dict(schema_version=1,id=identifier,signature=self.signature,started_at=agora(),status='running',
                                     identity=identity,environment=self.environment,stages={},blockchain_coverage_verified=False)
                escrever_json(manifest,self.manifest)
        except BaseException:
            self.lock.unlink(missing_ok=True)
            raise

    def hash(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        key = (str(path),stat.st_size,stat.st_mtime_ns)
        if key not in self.cache:
            self.cache[key] = sha256(path)
        return self.cache[key]

    def contrato(self, inputs):
        return assinatura(dict(execution_signature=self.signature,inputs={str(Path(p).resolve()):self.hash(p) for p in inputs}))

    def etapa(self, key, inputs, outputs, action):
        outputs = [Path(p).resolve() for p in outputs]
        if any(not p.is_relative_to(self.path) for p in outputs):
            raise ValueError('Etapa tentou escrever fora da execução.')
        contract = self.contrato(inputs)
        marker = self.path/'.etapas'/f'{key}.json'
        if marker.exists():
            prior = ler_json(marker)
            if prior['contract'] != contract:
                compatible = self.manifest.get('compatible_stage_signatures', [])
                accepted = any(prior['contract'] == assinatura(dict(execution_signature=sig,
                    inputs={str(Path(p).resolve()):self.hash(p) for p in inputs})) for sig in compatible)
                if not accepted:
                    raise ValueError(f'Entradas alteradas em {key}; use outra execução.')
            if prior['status'] == 'completed':
                if not all(p.is_file() and self.hash(p)==prior['outputs'][str(p)] for p in outputs):
                    raise ValueError(f'Saída concluída alterada ou ausente: {key}')
                return prior['result']
        elif any(p.exists() for p in outputs):
            raise ValueError(f'Saída existente sem registro de propriedade: {key}')
        record = dict(key=key,contract=contract,status='running',started_at=agora(),
                      inputs={str(Path(p).resolve()):self.hash(p) for p in inputs})
        escrever_json(marker,record)
        start = time.perf_counter()
        print(f'Etapa {key}',flush=True)
        try:
            result = action()
            if not all(p.is_file() for p in outputs):
                raise ValueError(f'Saídas incompletas: {key}')
            if self.integrity.get('validar_saidas'):
                from .integridade import conferir, comparar
                if any(sha256(Path(p)) != digest for p,digest in record['inputs'].items()):
                    raise ValueError(f'Entrada mudou durante o cálculo: {key}')
                record['verification'] = [conferir(p) for p in outputs]
                record['comparisons'] = [item for p in outputs for item in
                    comparar(p, self.path, self.integrity.get('comparar_execucoes', []))]
                # O hash final é recalculado, sem aproveitar o cache de leitura.
                for p in outputs:
                    if sha256(p) != next(v['sha256'] for v in record['verification'] if v['path']==str(p)):
                        raise ValueError(f'Saída mudou durante a conferência: {p}')
            record.update(status='completed',result=result,outputs={str(p):self.hash(p) for p in outputs})
        except BaseException as exc:
            record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            process = psutil.Process()
            record.update(ended_at=agora(),elapsed_seconds=time.perf_counter()-start,process_memory_bytes=process.memory_info()._asdict())
            escrever_json(marker,record)
            self.manifest['stages'][key] = {k:v for k,v in record.items() if k not in ('inputs','outputs','result')}
            self.manifest['updated_at'] = agora()
            escrever_json(self.path/'manifesto.json',self.manifest)
        return result

    def finalizar(self,status):
        self.manifest.update(status=status,updated_at=agora())
        self.manifest['artifacts'] = {path:digest for marker in (self.path/'.etapas').glob('*.json')
                                      for path,digest in ler_json(marker).get('outputs',{}).items()}
        self.manifest['artifacts'].update(self.manifest.get('correlation_outputs',{}))
        self.manifest['artifacts'].update(self.manifest.get('intersection_outputs',{}))
        if status == 'completed':
            self.manifest['completed_at'] = agora()
        escrever_json(self.path/'manifesto.json',self.manifest)

    def close(self):
        self.lock.unlink(missing_ok=True)

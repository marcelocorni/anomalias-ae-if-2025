"""Importação conferida de etapas independentes dos modelos."""
from __future__ import annotations

import importlib.metadata
import hashlib
import platform
import shutil
import time
import traceback
from pathlib import Path

import pyarrow.parquet as pq

from .atributos import FEATURES,METADATA
from .comum import agora,assinatura,escrever_json,ler_json,sha256
from .execucao import Execucao,codigo

CORRELATION_FILES=[f'correlacao-{method}-treino.{ext}'
                   for method in ('pearson','spearman') for ext in ('pdf','png')]+['correlacao-treino.json']
DATA_CODE=('atributos','auditoria','preprocessamento','comum','dados','recortes','labelcloud','drift')


def _same_code(recorded,root,names):
    normalized={name.replace('\\','/'):digest for name,digest in recorded.items()}
    for name in names:
        relative=f'src/{name}.py'
        if normalized.get(relative)!=sha256(Path(root)/relative):
            raise ValueError(f'Código incompatível para reaproveitamento: {relative}')


def _same_packages(recorded,names):
    for name in names:
        if recorded.get(name)!=importlib.metadata.version(name):
            raise ValueError(f'Versão incompatível para reaproveitamento: {name}')


def planejar(root,source,config,paths,identifier,records,mode='completo'):
    root,source=Path(root).resolve(),Path(source).resolve()
    target=(paths['resultados']/identifier).resolve()
    if source==target or source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError('Origem e destino devem ser separados.')
    if not target.is_relative_to(paths['resultados'].resolve()) or target==paths['resultados'].resolve():
        raise ValueError('Destino fora dos resultados.')
    for name in ('.lock','operacao/.supervisor.lock'):
        if (source/name).exists() or (target/name).exists():
            raise RuntimeError('Há um bloqueio na origem ou no destino; aguarde o término da execução.')
    manifest_path=source/'manifesto.json'
    manifest=ler_json(manifest_path)
    manifest_hash=sha256(manifest_path)
    if manifest['status']!='completed' or manifest.get('scientific_status')!='completed':
        raise ValueError('A origem precisa estar integralmente concluída.')
    identity=manifest['identity']
    if assinatura(identity)!=manifest['signature']:
        raise ValueError('Identidade da origem inconsistente.')
    if any(not cfg.get('integridade',{}).get(flag) for cfg in (config,identity['config'])
           for flag in ('validar_entradas','validar_saidas')):
        raise ValueError('O reaproveitamento requer conferência de entradas e saídas nas duas execuções.')
    if identity['days']!=[r['day'] for r in records] or identity['mode']!=mode:
        raise ValueError('Dias ou modo diferentes da origem.')
    for key in ('schema_version','matriz','percentual_blocos','timezone_dados','janelas','labelcloud','drift'):
        if identity['config'][key]!=config[key]:
            raise ValueError(f'Configuração incompatível para reaproveitamento: {key}')
    if Path(identity['paths']['dados']).resolve()!=paths['dados'].resolve():
        raise ValueError('Caminho dos dados originais diferente da origem.')
    if Path(identity['paths']['labelcloud_csv']).resolve()!=paths['labelcloud_csv'].resolve():
        raise ValueError('Caminho do Label Cloud diferente da origem.')
    if identity['python']!=platform.python_version():
        raise ValueError('Versão do Python diferente da origem.')
    _same_code(identity['code'],root,DATA_CODE)
    _same_packages(identity['packages'],('duckdb','numpy','pyarrow'))
    logical=(Path(identity['paths']['resultados'])/manifest['id']).resolve()
    artifacts={Path(p).resolve():digest for p,digest in manifest['artifacts'].items()}
    stages=[];external={};source_records={};rows_by_day={}

    def physical(relative):
        path=(source/relative).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError(f'Artefato ausente ou fora da origem: {relative}')
        return path

    def add(key,inputs,outputs):
        marker=physical(Path('.etapas')/f'{key}.json')
        prior=ler_json(marker)
        if prior['status']!='completed' or prior['key']!=key:
            raise ValueError(f'Etapa de origem incompleta: {key}')
        signatures=[manifest['signature'],*manifest.get('compatible_stage_signatures',[])]
        if not any(prior['contract']==assinatura(dict(execution_signature=s,inputs=prior['inputs'])) for s in signatures):
            raise ValueError(f'Contrato de origem inconsistente: {key}')
        old_inputs={Path(p).resolve():digest for p,digest in prior['inputs'].items()}
        if set(old_inputs)!=set(inputs):
            raise ValueError(f'Entradas inesperadas na etapa de origem: {key}')
        new_inputs={}
        for path,digest in old_inputs.items():
            if path.is_relative_to(logical):
                if artifacts.get(path)!=digest:raise ValueError(f'Entrada derivada inconsistente: {key}')
                new_inputs[str(target/path.relative_to(logical))]=digest
            else:
                if path in external and external[path]!=digest:raise ValueError('Entrada original com hashes divergentes.')
                external[path]=digest;new_inputs[str(path)]=digest
        expected={str(logical/relative):artifacts.get(logical/relative) for relative in outputs}
        if any(d is None for d in expected.values()) or prior['outputs']!=expected:
            raise ValueError(f'Saídas de origem inconsistentes: {key}')
        files=[dict(relative=str(relative),source=str(physical(relative)),sha256=expected[str(logical/relative)],
                    bytes=physical(relative).stat().st_size) for relative in outputs]
        spec=dict(key=key,inputs=new_inputs,files=files,result=prior['result'],
                  source_record=str(marker),source_record_sha256=sha256(marker),source_contract=prior['contract'])
        stages.append(spec);source_records[str(marker)]=spec['source_record_sha256']
        return prior

    raw={};processed={}
    for r in records:
        day,window=r['day'],r['window']
        prior=add('verificar-entrada-'+day,[Path(p).resolve() for p in r['files']],
                  [Path('integridade-entradas')/f'{day}.json'])
        rows_by_day[day]=prior['result']['rows']
    for r in records:
        day,window=r['day'],r['window'];relative=Path('atributos')/window/f'{day}.parquet'
        prior=add('atributos-'+day,[Path(p).resolve() for p in r['files']],[relative])
        table=pq.ParquetFile(physical(relative))
        if (table.metadata.num_rows!=rows_by_day[day] or prior['result']['rows']!=rows_by_day[day]
            or table.schema_arrow.names!=METADATA+FEATURES):
            raise ValueError(f'População ou esquema de atributos inconsistente: {day}')
        raw.setdefault(window,[]).append(logical/relative)
    all_raw=[p for files in raw.values() for p in files]
    add('auditoria-global',all_raw,[Path('auditoria-global.json')])
    train=next(w['chave'] for w in config['janelas'] if w['papel']=='train')
    scaler_stage=add('ajustar-preprocessamento',raw[train],[Path('preprocessamento.json')])
    scaler=ler_json(physical('preprocessamento.json'))
    if (scaler['features']!=FEATURES or scaler['fit_role']!='train'
        or scaler['input_signature']!=scaler_stage['contract']
        or assinatura({k:v for k,v in scaler.items() if k!='signature'})!=scaler['signature']):
        raise ValueError('Contrato do scaler inconsistente.')
    for r in records:
        day,window=r['day'],r['window'];relative=Path('preprocessado')/window/f'{day}.parquet'
        prior=add('transformar-'+day,[logical/'atributos'/window/f'{day}.parquet',logical/'preprocessamento.json'],[relative])
        table=pq.ParquetFile(physical(relative))
        if (table.metadata.num_rows!=rows_by_day[day] or prior['result']['rows']!=rows_by_day[day]
            or table.schema_arrow.names!=['row_id']+FEATURES
            or prior['result']['preprocessing_signature']!=scaler['signature']):
            raise ValueError(f'População ou esquema pré-processado inconsistente: {day}')
        processed.setdefault(window,[]).append(logical/relative)
    add('labelcloud-normalizar',[paths['labelcloud_csv'].resolve()],[Path('labelcloud.parquet')])
    add('drift-B',[p for files in processed.values() for p in files],[Path('drift-B.json')])

    corr_path=physical('.correlacao-treino/manifesto.json');corr=ler_json(corr_path);ci=corr['identity']
    if (corr['status']!='completed' or corr['signature']!=assinatura(ci)
        or ci['source_signature']!=manifest['signature'] or ci['features']!=FEATURES
        or ci['representation']!='preprocessado_float64' or ci['methods']!=['pearson','spearman_average_rank']
        or ci['code_sha256']!=sha256(root/'src/correlacao.py')
        or ci['rows']!=sum(rows_by_day[r['day']] for r in records if r['window']==train)
        or {Path(p).resolve():d for p,d in ci['inputs'].items()}!={p:artifacts[p] for p in processed[train]}):
        raise ValueError('Contrato das correlações incompatível.')
    _same_code(ci['helpers'],root,('comum','atributos'))
    _same_packages(ci['packages'],('duckdb','numpy','pyarrow','matplotlib'))
    expected_corr={str(logical/'figuras'/name):artifacts.get(logical/'figuras'/name) for name in CORRELATION_FILES}
    if any(d is None for d in expected_corr.values()) or corr['outputs']!=expected_corr:
        raise ValueError('Saídas das correlações inconsistentes.')
    source_records[str(corr_path)]=sha256(corr_path)
    stages.append(dict(key='correlacoes-reutilizadas',inputs={str(target/p.relative_to(logical)):artifacts[p] for p in processed[train]},
        files=[dict(relative=str(Path('figuras')/name),source=str(physical(Path('figuras')/name)),
                    sha256=expected_corr[str(logical/'figuras'/name)],bytes=physical(Path('figuras')/name).stat().st_size)
               for name in CORRELATION_FILES],result=dict(status='completed',reused=True),
        source_record=str(corr_path),source_record_sha256=sha256(corr_path),source_contract=corr['signature']))
    transfer_identity=dict(source=str(source),source_manifest_sha256=manifest_hash,source_signature=manifest['signature'],
                           source_records=source_records,config=config,paths={k:str(v) for k,v in paths.items()},
                           identifier=identifier,days=[r['day'] for r in records],mode=mode,code=codigo(root))
    if target.exists() and (target/'manifesto.json').exists():
        prior=ler_json(target/'manifesto.json')
        if prior.get('reuse',{}).get('contract')!=assinatura(transfer_identity):
            raise ValueError('Destino já utilizado por outra execução ou importação.')
    elif target.exists() and any(p.name!='operacao' for p in target.iterdir()):
        raise ValueError('Destino não está vazio.')
    files=[f for stage in stages for f in stage['files']]
    remaining=sum(f['bytes'] for f in files if not (target/f['relative']).exists())
    temporary=max(f['bytes'] for f in files)
    free=shutil.disk_usage(root).free
    if free-remaining-temporary<config['recursos']['min_free_gib']*2**30:
        raise ValueError(f'A cópia precisa de {remaining/2**30:.1f} GiB além da reserva de disco configurada.')
    return dict(root=root,source=source,target=target,identity=transfer_identity,stages=stages,external=external,
                source_manifest=manifest_path,source_manifest_sha256=manifest_hash,days=len(records),
                rows=sum(rows_by_day.values()),files=len(files),bytes=sum(f['bytes'] for f in files),
                remaining_bytes=remaining,temporary_bytes=temporary,
                free_gib=free/2**30,free_after_copy_gib=(free-remaining)/2**30)


def _copy_verified(source,target,digest):
    source,target=Path(source),Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+'.reuse.partial')
    copied=hashlib.sha256()
    with source.open('rb') as reader,temporary.open('wb') as writer:
        for chunk in iter(lambda:reader.read(8*1024*1024),b''):
            copied.update(chunk);writer.write(chunk)
    if copied.hexdigest()!=digest:raise ValueError(f'Arquivo de origem alterado: {source}')
    if sha256(temporary)!=digest:raise ValueError(f'Cópia com hash divergente: {target}')
    temporary.replace(target)


def executar(plan,config,paths,identifier,mode='completo'):
    source,target=plan['source'],plan['target'];identity=plan['identity'];contract=assinatura(identity)
    ctx=Execucao(plan['root'],paths['resultados'],identifier,config,paths,identity['days'],mode)
    state=dict(status='running',contract=contract,source=str(source),source_signature=identity['source_signature'],
               started_at=agora(),completed_stages=0,total_stages=len(plan['stages']))
    folder=target/'.reuso';start=time.perf_counter()
    try:
        if ctx.manifest.get('reuse',{}).get('contract') not in (None,contract):
            raise ValueError('Contrato de importação diferente do destino.')
        ctx.manifest['reuse']=state
        escrever_json(target/'manifesto.json',ctx.manifest)
        escrever_json(folder/'manifesto.json',dict(**state,identity=identity))
        if sha256(plan['source_manifest'])!=plan['source_manifest_sha256']:
            raise ValueError('Manifesto da origem mudou.')
        for index,(path,digest) in enumerate(plan['external'].items(),1):
            if ctx.hash(path)!=digest:raise ValueError(f'Entrada original alterada: {path}')
            if index%20==0 or index==len(plan['external']):
                print(f'Entradas originais conferidas: {index}/{len(plan["external"])}',flush=True)
        for index,stage in enumerate(plan['stages'],1):
            record_path=Path(stage['source_record'])
            if sha256(record_path)!=stage['source_record_sha256']:
                raise ValueError(f'Registro da origem alterado: {record_path}')
            for path,digest in stage['inputs'].items():
                if ctx.hash(path)!=digest:raise ValueError(f'Entrada incompatível: {path}')
            outputs=[target/f['relative'] for f in stage['files']]
            def copy(stage=stage):
                for f in stage['files']:
                    _copy_verified(f['source'],target/f['relative'],f['sha256'])
                return stage['result']
            ctx.etapa(stage['key'],list(stage['inputs']),outputs,copy)
            if any(ctx.hash(target/f['relative'])!=f['sha256'] for f in stage['files']):
                raise ValueError(f'Saída importada divergente: {stage["key"]}')
            marker=target/'.etapas'/f"{stage['key']}.json"
            record=ler_json(marker)
            record['reuse']=dict(source=str(source),source_signature=identity['source_signature'],
                source_record=str(record_path.relative_to(source)),source_record_sha256=stage['source_record_sha256'],
                source_contract=stage['source_contract'])
            escrever_json(marker,record)
            ctx.manifest['stages'][stage['key']]['reused']=True
            state['completed_stages']=index
            escrever_json(folder/'manifesto.json',dict(**state,identity=identity))
            if index%25==0 or index==len(plan['stages']):
                print(f'Etapas importadas e conferidas: {index}/{len(plan["stages"])}',flush=True)
        if sha256(plan['source_manifest'])!=plan['source_manifest_sha256']:
            raise ValueError('Manifesto da origem mudou durante a importação.')
        state.update(status='completed',ended_at=agora(),elapsed_seconds=time.perf_counter()-start,
                     files=plan['files'],bytes=plan['bytes'])
        escrever_json(folder/'manifesto.json',dict(**state,identity=identity))
        ctx.finalizar('partial')
        print('Importação concluída. Nenhum treinamento ou pipeline foi iniciado.',flush=True)
        return target
    except BaseException as exc:
        state.update(status='failed',ended_at=agora(),error=f'{type(exc).__name__}: {exc}')
        escrever_json(folder/'manifesto.json',dict(**state,identity=identity))
        ctx.finalizar('failed')
        with (folder/'erros.log').open('a',encoding='utf8') as stream:
            stream.write(agora()+'\n'+traceback.format_exc()+'\n')
        raise
    finally:
        ctx.close()

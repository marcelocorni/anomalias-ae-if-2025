"""Contagens exatas de sinalizações associadas a mev_bot, por dia e recorte."""
import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import psutil
import pyarrow.parquet as pq
from src.comum import conectar,parquet
from src.correlacao import Steps,now,read,sha,signature,write
from src.labelcloud import criar_tags

QUANTILES=('q990','q995','q999')


def contar_dia(con,attributes,ae,iff,tags,thresholds):
    criar_tags(con,[tags])
    con.execute(f'''CREATE OR REPLACE TEMP TABLE mev_daily AS
        SELECT m.row_id FROM {parquet([attributes])} m
        JOIN tags t ON lower(m.from_address)=t.address
        WHERE t.is_mev_bot''')
    positives,unique=con.execute('SELECT count(*),count(DISTINCT row_id) FROM mev_daily').fetchone()
    if positives!=unique:raise ValueError('Chaves ou endereços duplicados na associação diária.')
    expressions=['count(*)','count(DISTINCT m.row_id)']
    for key in QUANTILES:
        a=f"a.score >= {thresholds['AE'][key]:.17g}";b=f"b.score >= {thresholds['IF'][key]:.17g}"
        expressions += [f'count(*) FILTER(WHERE {a})',f'count(*) FILTER(WHERE {b})',f'count(*) FILTER(WHERE {a} AND {b})']
    result=con.execute(f'''SELECT {','.join(expressions)} FROM mev_daily m
        JOIN {parquet([ae])} a USING(row_id)
        JOIN {parquet([iff])} b USING(row_id)''').fetchone()
    if result[0]!=positives or result[1]!=positives:raise ValueError('Alinhamento dos escores MEV inconsistente.')
    thresholds_result={}
    for i,key in enumerate(QUANTILES):
        a,b,both=map(int,result[2+i*3:5+i*3]);union=a+b-both
        thresholds_result[key]=dict(AE=a,IF=b,intersection=both,union=union,AE_only=a-both,
                                    IF_only=b-both,neither=positives-union)
    return dict(associated=int(positives),thresholds=thresholds_result)


def reconciliar(actual,ae,iff):
    if actual['associated']!=ae['positives'] or ae['positives']!=iff['positives']:
        raise ValueError('Total associado diverge dos relatórios existentes.')
    for key in QUANTILES:
        row=actual['thresholds'][key]
        if any(type(row[name]) is not int or row[name]<0 for name in
               ('AE','IF','intersection','union','AE_only','IF_only','neither')):
            raise ValueError('Contagens de interseção inválidas.')
        if row['AE']!=ae['thresholds'][key]['tp'] or row['IF']!=iff['thresholds'][key]['tp']:
            raise ValueError(f'Contagens por modelo divergem dos relatórios: {key}.')
        if (not 0<=row['intersection']<=min(row['AE'],row['IF'])
            or row['AE_only']+row['intersection']!=row['AE']
            or row['IF_only']+row['intersection']!=row['IF']
            or row['AE_only']+row['intersection']+row['IF_only']!=row['union']
            or row['union']+row['neither']!=actual['associated']):
            raise ValueError('Interseção ou união inconsistente.')


def planejar(run,root=ROOT):
    run=Path(run).resolve();manifest=read(run/'manifesto.json')
    if manifest.get('scientific_status',manifest['status'])!='completed':
        raise ValueError('Os cálculos de origem precisam estar concluídos.')
    if not run.is_relative_to(Path(root).resolve()/'resultados'):raise ValueError('Origem fora de resultados.')
    windows=manifest['identity']['config']['janelas'];days=manifest['identity']['days']
    thresholds={m:read(run/f'limiares-{m}.json')['thresholds'] for m in ('AE','IF')}
    tags=run/'labelcloud.parquet';records=[]
    for day in days:
        window=next(w['chave'] for w in windows if w['inicio']<=day<w['fim_exclusivo'])
        files=[run/'atributos'/window/f'{day}.parquet',*[run/'escores'/m/window/f'{day}.parquet' for m in ('AE','IF')]]
        counts=[pq.ParquetFile(p).metadata.num_rows for p in files]
        if len(set(counts))!=1:raise ValueError(f'Populações diárias divergentes: {day}')
        records.append(dict(day=day,window=window,files=files,rows=counts[0]))
    windows=[w for w in windows if any(r['window']==w['chave'] for r in records)]
    reports={w['chave']:{m:read(run/'labelcloud'/f"{w['chave']}-{m}.json") for m in ('AE','IF')} for w in windows}
    inputs=[tags,*[run/f'limiares-{m}.json' for m in ('AE','IF')]]
    inputs += [run/'labelcloud'/f"{w['chave']}-{m}.json" for w in windows for m in ('AE','IF')]
    inputs += [p for record in records for p in record['files']]
    expected={str(p):manifest['artifacts'][str(p)] for p in inputs}
    identity=dict(source_signature=manifest['signature'],inputs=expected,thresholds=thresholds,
                  code_sha256=sha(Path(__file__)),helpers={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/comum.py',ROOT/'src/correlacao.py',ROOT/'src/labelcloud.py']},
                  packages={p:importlib.metadata.version(p) for p in ('duckdb','pyarrow','psutil')})
    return manifest,records,reports,tags,thresholds,identity


def identidade_retomada(identity,previous,continuation):
    if previous['signature']!=signature(previous['identity']):
        raise ValueError('Manifesto das interseções inconsistente.')
    if previous['signature']==signature(identity):return identity
    accepted=read(continuation) if Path(continuation).exists() else {}
    if (accepted.get('source_signature')!=previous['signature']
        or accepted.get('identity')!=identity or accepted.get('signature')!=signature(identity)):
        raise ValueError('Entradas, ambiente ou código mudaram; preserve os checkpoints.')
    # A autorização identifica exatamente o código e as entradas revisados.
    return previous['identity']


@contextmanager
def lock_run(run,already_locked=False):
    lock=run/'.lock'
    if already_locked:
        owned=read(lock) if lock.exists() else {}
        if owned!={'pid':os.getpid(),'created':psutil.Process().create_time()}:
            raise RuntimeError('O processo não possui o bloqueio da execução.')
        yield
        return
    if lock.exists():
        previous=read(lock)
        try:active=psutil.Process(previous['pid']).create_time()==previous['created']
        except psutil.NoSuchProcess:active=False
        if active:raise RuntimeError('Outra execução está ativa; aguarde seu término.')
        lock.unlink()
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as stream:json.dump(dict(pid=os.getpid(),created=psutil.Process().create_time()),stream)
    try:yield
    finally:lock.unlink(missing_ok=True)


def executar(run,plan,already_locked=False):
    run=Path(run).resolve()
    manifest,records,reports,tags,thresholds,identity=plan
    folder=run/'.intersecao-mev';folder.mkdir(exist_ok=True)
    state=folder/'manifesto.json'
    previous=read(state) if state.exists() else None
    if previous is not None:identity=identidade_retomada(identity,previous,folder/'continuidade.json')
    contract=signature(identity)
    record=dict(signature=contract,identity=identity,status='running',started_at=now())
    with lock_run(run,already_locked):
        if previous is not None and previous['status']=='completed':
            for path,digest in identity['inputs'].items():
                if not Path(path).is_file() or sha(path)!=digest:raise ValueError(f'Entrada alterada: {path}')
            target=run/'avaliacao/intersecao-mev.json'
            if (set(previous.get('outputs',{}))!={str(target)}
                or not target.is_file() or sha(target)!=previous['outputs'][str(target)]):
                raise ValueError('Saída de interseções concluída alterada ou ausente.')
            document=read(target)
            if (document['status']!='completed' or document['analysis_signature']!=contract
                or document['source_signature']!=manifest['signature'] or set(document['windows'])!=set(reports)):
                raise ValueError('Proveniência das interseções inconsistente.')
            for window,actual in document['windows'].items():
                reconciliar(actual,reports[window]['AE'],reports[window]['IF'])
                if actual['rows']!=reports[window]['AE']['rows']:raise ValueError('População do recorte divergente.')
            print('Interseções MEV concluídas e conferidas; nenhum cálculo repetido.',flush=True)
            return previous
        started=time.perf_counter()
        write(state,record)
        try:
            steps=Steps(folder,contract,reserve_gib=16)
            common=[tags,*[run/f'limiares-{m}.json' for m in ('AE','IF')]]
            common += [run/'labelcloud'/f'{w}-{m}.json' for w in reports for m in ('AE','IF')]
            for p in common:
                if sha(p)!=identity['inputs'][str(p)]:raise ValueError(f'Entrada alterada: {p}')
            results=[]
            with conectar(folder/'duckdb','8GB',4) as con:
                con.execute("SET max_temp_directory_size='16GB'")
                criar_tags(con,[tags])
                n,unique=con.execute('SELECT count(*),count(DISTINCT address) FROM tags').fetchone()
                if not n or n!=unique:raise ValueError('Referência MEV agrupada vazia ou inconsistente.')
                for i,item in enumerate(records,1):
                    for p in item['files']:
                        if sha(p)!=identity['inputs'][str(p)]:raise ValueError(f'Entrada alterada: {p}')
                    output=folder/'dias'/f"{item['day']}.json"
                    def action(item=item,output=output):
                        result=contar_dia(con,*item['files'],tags,thresholds)
                        for p in item['files']:
                            if sha(p)!=identity['inputs'][str(p)]:raise ValueError(f'Entrada mudou durante a contagem: {p}')
                        result.update(day=item['day'],window=item['window'],rows=item['rows'])
                        write(output,result);return result
                    result=steps.run(item['day'],[output],action);results.append(result)
                    print(f"Dias conferidos: {i}/{len(records)}",flush=True)
            windows={}
            for window in reports:
                subset=[r for r in results if r['window']==window]
                aggregate=dict(rows=sum(r['rows'] for r in subset),associated=sum(r['associated'] for r in subset),thresholds={})
                for key in QUANTILES:
                    aggregate['thresholds'][key]={name:sum(r['thresholds'][key][name] for r in subset)
                        for name in ('AE','IF','intersection','union','AE_only','IF_only','neither')}
                reconciliar(aggregate,reports[window]['AE'],reports[window]['IF'])
                if aggregate['rows']!=reports[window]['AE']['rows']:raise ValueError('População do recorte divergente.')
                windows[window]=aggregate
            for p in common:
                if sha(p)!=identity['inputs'][str(p)]:raise ValueError(f'Entrada de referência mudou: {p}')
            target=run/'avaliacao/intersecao-mev.json'
            document=dict(status='completed',source_signature=manifest['signature'],analysis_signature=contract,
                          association='lower(from_address)=address AND is_mev_bot',windows=windows)
            steps.run('agregar',[target],lambda:(write(target,document) or dict(windows=len(windows))))
            record.update(status='completed',ended_at=now(),elapsed_seconds=time.perf_counter()-started,
                          outputs={str(target):sha(target)})
            write(state,record)
            print(json.dumps(dict(output=str(target),q995={w:v['thresholds']['q995'] for w,v in windows.items()}),indent=2),flush=True)
            return record
        except BaseException as exc:
            record.update(status='failed',ended_at=now(),error=f'{type(exc).__name__}: {exc}');write(state,record)
            with (folder/'erros.log').open('a',encoding='utf8') as stream:stream.write(now()+'\n'+traceback.format_exc()+'\n')
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origem',type=Path,default=ROOT/'resultados/anual-2025-b-03')
    parser.add_argument('--executar',action='store_true');args=parser.parse_args()
    run=args.origem.resolve();plan=planejar(run)
    print(json.dumps(dict(days=len(plan[1]),rows=sum(r['rows'] for r in plan[1]),thresholds=list(QUANTILES),
                          retrain=False,execution_started=False),indent=2),flush=True)
    if args.executar:executar(run,plan)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'Contagem interrompida: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        raise SystemExit(1)

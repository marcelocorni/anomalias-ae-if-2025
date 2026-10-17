"""Correlações exatas no treino, com processamento por lotes e retomada."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import traceback
import time

import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.atributos import FEATURES
from src.comum import conectar, ident, parquet


def now(): return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf8'))


def replace_file(temporary,path):
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError as exc:
            if getattr(exc,'winerror',None) not in (5,32,33) or attempt==4:raise
            time.sleep((.2,.5,1.,2.)[attempt])


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    replace_file(temporary,path)


def signature(data): return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def resume_identity(current,prior):
    if prior['signature']!=signature(prior['identity']):raise ValueError('Manifesto inconsistente.')
    if prior['signature']==signature(current):return current
    previous=dict(prior['identity']);updated=dict(current)
    previous_code=previous.pop('code_sha256');updated.pop('code_sha256')
    if previous_code!='1a13e2e1ba75279f0a7943163bf2229885e7cdf33361bc1f2ff0027e8fabe0bc' or previous!=updated:
        raise ValueError('Código, entrada ou parâmetros mudaram; preserve a pasta de checkpoints.')
    return prior['identity']


def merge(a,b):
    if a is None: return {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in b.items()}
    n=a['n']+b['n'];delta=b['mean']-a['mean']
    return dict(n=n,mean=a['mean']+delta*(b['n']/n),
                m2=a['m2']+b['m2']+np.outer(delta,delta)*(a['n']*b['n']/n),
                low=np.minimum(a['low'],b['low']),high=np.maximum(a['high'],b['high']))


def moments(values):
    if values.ndim!=2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Lote vazio, nulo ou não finito.')
    mean=values.mean(axis=0,dtype=np.float64)
    centered=values-mean
    result=dict(n=len(values),mean=mean,m2=centered.T@centered,
                low=values.min(axis=0),high=values.max(axis=0))
    if not np.isfinite(result['m2']).all():raise ValueError('Covariância não finita.')
    return result


def coefficients(state):
    variance=np.diag(state['m2']);constant=state['low']==state['high']
    if (variance<0).any() or not np.allclose(state['m2'],state['m2'].T,rtol=1e-12,atol=1e-12):
        raise ValueError('Covariância inconsistente.')
    denominator=np.sqrt(np.outer(variance,variance))
    corr=np.divide(state['m2'],denominator,out=np.full_like(state['m2'],np.nan),where=denominator>0)
    corr[constant,:]=np.nan;corr[:,constant]=np.nan
    finite=corr[np.isfinite(corr)]
    if finite.size and np.max(np.abs(finite))>1+1e-8:raise ValueError('Correlação fora de [-1,1].')
    corr=np.clip(corr,-1,1)
    return corr,constant


class Steps:
    def __init__(self,folder,contract,reserve_gib=32):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.contract=contract;self.reserve=reserve_gib*2**30

    def guard(self):
        if shutil.disk_usage(self.folder).free<self.reserve:
            raise ValueError('Espaço livre abaixo da reserva; preserve os checkpoints.')

    def run(self,key,outputs,action):
        outputs=[Path(p) for p in outputs];marker=self.folder/'etapas'/f'{key}.json'
        partial=marker.with_name(marker.name+'.partial')
        if partial.exists():
            try:pending=read(partial)
            except json.JSONDecodeError:pending={}
            if (pending.get('contract')==self.contract and pending.get('status')=='completed'
                and set(pending.get('outputs',{}))=={str(p) for p in outputs}
                and all(p.is_file() and sha(p)==pending['outputs'][str(p)] for p in outputs)):
                replace_file(partial,marker)
                print(f'Checkpoint recuperado: {key}',flush=True)
        if marker.exists():
            prior=read(marker)
            if prior['contract']!=self.contract:raise ValueError('Checkpoint de outro contrato.')
            if prior['status']=='completed':
                if any(not p.is_file() or sha(p)!=prior['outputs'][str(p)] for p in outputs):
                    raise ValueError(f'Checkpoint alterado: {key}')
                return prior['result']
        elif any(p.exists() for p in outputs):
            raise ValueError(f'Artefato sem registro de propriedade: {key}')
        self.guard()
        record=dict(contract=self.contract,status='running',started_at=now())
        write(marker,record);print(f'Etapa {key}',flush=True)
        try:
            result=action()
            hashes={str(p):sha(p) for p in outputs}
            if any(sha(p)!=hashes[str(p)] for p in outputs):raise ValueError('Hash de saída inconsistente.')
            record.update(status='completed',result=result,outputs=hashes)
            return result
        except BaseException as exc:
            record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            record['ended_at']=now();write(marker,record)


def save_state(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial')
    with temporary.open('wb') as stream:np.savez(stream,**state)
    temporary.replace(path)


def load_state(path):
    with np.load(path,allow_pickle=False) as arrays:
        return {k:int(arrays[k]) if k=='n' else arrays[k].copy() for k in arrays.files}


def map_ranks(con,files,feature,target):
    """Posto médio = frequência acumulada - (frequência do valor - 1)/2."""
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    column=ident(feature)
    query=f'''SELECT value,CAST(sum(frequency) OVER(ORDER BY value ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        -(frequency-1)/2.0 AS DOUBLE) AS average_rank
        FROM (SELECT {column} AS value,count(*) AS frequency FROM {parquet(files)} GROUP BY {column})
        ORDER BY value'''
    temporary=target.with_name(target.name+'.partial')
    from src.comum import q
    con.execute(f'COPY ({query}) TO {q(temporary.as_posix())} (FORMAT PARQUET,COMPRESSION ZSTD)')
    temporary.replace(target)
    source=pq.ParquetFile(target);count=source.metadata.num_rows
    outputs=[target.with_suffix('.values.npy'),target.with_suffix('.ranks.npy')]
    temps=[p.with_name(p.name+'.partial') for p in outputs]
    arrays=[np.lib.format.open_memmap(p,mode='w+',dtype=np.float64,shape=(count,)) for p in temps]
    offset=0;previous=None
    try:
        for batch in source.iter_batches(batch_size=131072,use_threads=False):
            values=batch.column(0).to_numpy(zero_copy_only=False)
            ranks=batch.column(1).to_numpy(zero_copy_only=False)
            if not np.isfinite(values).all() or not np.isfinite(ranks).all():raise ValueError('Mapa de postos inválido.')
            if (previous is not None and values[0]<=previous) or np.any(np.diff(values)<=0):
                raise ValueError('Mapa de postos sem ordenação estrita.')
            previous=values[-1]
            for a,v in zip(arrays,(values,ranks)):a[offset:offset+len(v)]=v
            offset+=len(values)
        if offset!=count:raise ValueError('Contagem do mapa de postos divergente.')
    finally:
        for a in arrays:a.flush();a._mmap.close()
    for a,b in zip(temps,outputs):a.replace(b)
    return dict(unique_values=count)


def rank_values(values,maps):
    values=np.array(values,dtype=np.float64,copy=True)
    for i,(keys,ranks) in enumerate(maps):
        positions=np.searchsorted(keys,values[:,i])
        if np.any(positions>=len(keys)) or not np.array_equal(keys[positions],values[:,i]):
            raise ValueError('Valor ausente no mapa de postos do treino.')
        values[:,i]=ranks[positions]
    return values


def calculate_day(source,target,batch_size,maps=None):
    state=None;file=pq.ParquetFile(source)
    for batch in file.iter_batches(batch_size=batch_size,columns=FEATURES,use_threads=False):
        values=np.column_stack([batch.column(i).to_numpy(zero_copy_only=False) for i in range(len(FEATURES))]).astype(np.float64,copy=False)
        if maps is not None:values=rank_values(values,maps)
        state=merge(state,moments(values))
    if state is None or state['n']!=file.metadata.num_rows:raise ValueError('População diária divergente.')
    save_state(target,state)
    return dict(rows=state['n'])


def calculate(files,steps,method,batch_size,maps=None):
    total=None
    for source in files:
        target=steps.folder/'momentos'/method/(source.stem+'.npz')
        steps.run(f'{method}-{source.stem}',[target],lambda s=source,p=target:calculate_day(s,p,batch_size,maps))
        total=merge(total,load_state(target))
    corr,constant=coefficients(total)
    return dict(method=method,rows=total['n'],constant_features=[f for f,c in zip(FEATURES,constant) if c],
                matrix=[[float(v) if np.isfinite(v) else None for v in row] for row in corr])


def plot(result,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'font.size':9})
    matrix=np.array([[np.nan if v is None else v for v in row] for row in result['matrix']])
    fig,ax=plt.subplots(figsize=(8.4,8.2),layout='constrained')
    cmap=plt.get_cmap('RdBu_r').with_extremes(bad='#d9dde2')
    image=ax.pcolormesh(range(len(FEATURES)+1),range(len(FEATURES)+1),
                        np.ma.masked_invalid(matrix),vmin=-1,vmax=1,cmap=cmap,
                        edgecolors='white',linewidth=.12,rasterized=False)
    ticks=np.arange(len(FEATURES))+.5
    ax.set(xticks=ticks,yticks=ticks,xlim=(0,len(FEATURES)),ylim=(len(FEATURES),0))
    ax.set_xticklabels(FEATURES,rotation=90,fontsize=7.5)
    ax.set_yticklabels(FEATURES,fontsize=7.5)
    ax.tick_params(length=0)
    title='Pearson' if result['method']=='pearson' else 'Spearman (postos médios)'
    subtitle='Prévia de layout - dados sintéticos' if result.get('preview') else 'Treino: janeiro a abril de 2025'
    ax.set_title(f'Correlação de {title}\n{subtitle}',fontsize=11)
    bar=fig.colorbar(image,ax=ax,shrink=.8,label='Coeficiente de correlação')
    if bar.solids is not None:bar.solids.set_rasterized(False)
    note=('Prévia técnica; não representa resultados do experimento.' if result.get('preview') else
          f"{result['rows']:,} transações | 21 atributos pré-processados | sem amostragem".replace(',','.'))
    if result['constant_features']:note+='\nCinza: correlação indefinida por atributo constante.'
    fig.supxlabel(note,fontsize=8)
    for ext in ('pdf','png'):fig.savefig(Path(folder)/f"correlacao-{result['method']}-treino.{ext}",bbox_inches='tight',dpi=200)
    plt.close(fig)


def prepare(run):
    run=Path(run).resolve();manifest=read(run/'manifesto.json')
    if manifest.get('scientific_status',manifest['status'])!='completed':raise ValueError('Os cálculos de origem precisam estar concluídos.')
    expected=[r for r in manifest['identity']['config']['janelas'] if r['papel']=='train']
    if len(expected)!=1:raise ValueError('Recorte de treino inesperado.')
    files=sorted((run/'preprocessado'/expected[0]['chave']).glob('*.parquet'))
    selected=[d for d in manifest['identity']['days'] if expected[0]['inicio']<=d<expected[0]['fim_exclusivo']]
    if [p.stem for p in files]!=selected:raise ValueError('Recorte de treino incompleto ou inesperado.')
    if any(not expected[0]['inicio']<=p.stem<expected[0]['fim_exclusivo'] for p in files):
        raise ValueError('Arquivo de outro recorte no treino.')
    count=sum(pq.ParquetFile(p).metadata.num_rows for p in files)
    training_stage=read(run/'.etapas/ajustar-if.json')
    if count!=training_stage['result']['training_rows_total']:raise ValueError('Contagem do treino divergente.')
    return run,manifest,files,count


def execute(run,manifest,files,rows,destination,memory,threads,batch_size):
    started=time.perf_counter()
    destination=Path(destination).resolve();folder=destination.parent/'.correlacao-treino'
    folder.mkdir(parents=True,exist_ok=True)
    import psutil,os
    lock=folder/'.lock'
    if lock.exists():
        record=read(lock)
        try:active=psutil.Process(record['pid']).create_time()==record['created']
        except psutil.NoSuchProcess:active=False
        if active:raise RuntimeError('Cálculo de correlação já ativo.')
        lock.unlink()
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as stream:json.dump(dict(pid=os.getpid(),created=psutil.Process().create_time()),stream)
    con=None;maps=[];record=None
    try:
        hashes={}
        for i,p in enumerate(files,1):
            a=sha(p)
            if a!=manifest['artifacts'].get(str(p)) or sha(p)!=a:raise ValueError(f'Entrada alterada ou leitura inconsistente: {p}')
            hashes[str(p)]=a
            if i%20==0:print(f'Entradas conferidas: {i}/{len(files)}',flush=True)
        identity=dict(source_signature=manifest['signature'],features=FEATURES,inputs=hashes,
                      rows=rows,representation='preprocessado_float64',methods=['pearson','spearman_average_rank'],
                      batch_size=batch_size,memory_limit=memory,threads=threads,
                      code_sha256=sha(Path(__file__)),helpers={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/comum.py',ROOT/'src/atributos.py']},
                      packages={p:importlib.metadata.version(p) for p in ['numpy','pyarrow','duckdb','matplotlib']})
        state=folder/'manifesto.json'
        prior=read(state) if state.exists() else None
        if prior is not None:identity=resume_identity(identity,prior)
        contract=signature(identity)
        if prior is not None and prior['status']=='completed':
            if not prior.get('outputs') or any(not Path(p).is_file() or sha(Path(p))!=digest for p,digest in prior['outputs'].items()):
                raise ValueError('Saída de correlação concluída alterada ou ausente.')
            print('Correlações concluídas e conferidas; nenhum cálculo repetido.',flush=True)
            return prior
        record=dict(signature=contract,identity=identity,implementation_sha256=sha(Path(__file__)),status='running',updated_at=now())
        write(state,record);steps=Steps(folder,contract)
        destination.mkdir(parents=True,exist_ok=True)
        con=conectar(folder/'duckdb',memory,threads)
        con.execute("SET max_temp_directory_size='64GB'")
        with threadpool_limits(limits=threads):
            pearson=calculate(files,steps,'pearson',batch_size)
            for feature in FEATURES:
                target=folder/'postos'/(feature+'.parquet')
                outputs=[target,target.with_suffix('.values.npy'),target.with_suffix('.ranks.npy')]
                steps.run('postos-'+feature,outputs,lambda f=feature,t=target:map_ranks(con,files,f,t))
                maps.append((np.load(outputs[1],mmap_mode='r'),np.load(outputs[2],mmap_mode='r')))
            spearman=calculate(files,steps,'spearman',batch_size,maps)
        if pearson['rows']!=rows or spearman['rows']!=rows:raise ValueError('Contagem total divergente.')
        targets=[destination/f'correlacao-{m}-treino.{ext}' for m in ('pearson','spearman') for ext in ('pdf','png')]
        targets.append(destination/'correlacao-treino.json')
        def figures():
            for result in (pearson,spearman):plot(result,destination)
            write(targets[-1],dict(features=FEATURES,pearson=pearson,spearman=spearman,source_signature=manifest['signature'],representation='preprocessado_float64'))
            return dict(pdf_count=2,png_count=2)
        steps.run('figuras',targets,figures)
        if any(sha(Path(p))!=digest for p,digest in hashes.items()):raise ValueError('Entrada mudou durante o cálculo.')
        record.update(status='completed',updated_at=now(),elapsed_seconds=time.perf_counter()-started,outputs={str(p):sha(p) for p in targets})
        write(state,record);write(destination/'correlacao-treino-manifesto.json',record)
        print(f'Concluído: {destination}',flush=True)
        return record
    except BaseException as exc:
        if record is not None:
            record.update(status='failed',updated_at=now(),error=f'{type(exc).__name__}: {exc}');write(folder/'manifesto.json',record)
        with (folder/'erros.log').open('a',encoding='utf8') as stream:stream.write(now()+'\n'+traceback.format_exc()+'\n')
        raise
    finally:
        for values,ranks in maps:values._mmap.close();ranks._mmap.close()
        if con is not None:con.close()
        lock.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description='Pearson e Spearman exatos dos 21 atributos no treino completo.')
    parser.add_argument('--origem',type=Path,default=ROOT/'resultados/anual-2025-b-03')
    parser.add_argument('--saida',type=Path)
    parser.add_argument('--executar',action='store_true')
    args=parser.parse_args();run,manifest,files,rows=prepare(args.origem)
    destination=(args.saida or run/'figuras').resolve()
    if destination!=run/'figuras':raise ValueError('As figuras devem ficar na própria execução.')
    print(json.dumps(dict(rows=rows,days=len(files),features=21,methods=['Pearson','Spearman'],sampled=False,
                          destination=str(destination),execution_started=False),ensure_ascii=False,indent=2),flush=True)
    if args.executar:execute(run,manifest,files,rows,destination,'16GB',8,131072)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'Cálculo interrompido: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        raise SystemExit(1)

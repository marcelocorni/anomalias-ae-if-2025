import copy
import importlib.metadata
import platform
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src import atributos,auditoria,correlacao,drift,execucao,integridade,labelcloud,pipeline,preprocessamento,reutilizacao
from src.comum import conectar,escrever_json,ler_json,sha256
from src.configuracao import RAIZ


@pytest.fixture
def preparation(tmp_path,config,raw_factory,monkeypatch):
    root=tmp_path/'project';root.mkdir();(root/'src').mkdir()
    for name in ('pyproject.toml','uv.lock'):
        shutil.copy2(RAIZ/name,root/name)
    for name in (*reutilizacao.DATA_CODE,'correlacao'):
        shutil.copy2(RAIZ/f'src/{name}.py',root/f'src/{name}.py')
    environment=dict(python=platform.python_version(),packages={name:importlib.metadata.version(name)
        for name in ('duckdb','numpy','pyarrow','matplotlib')})
    monkeypatch.setattr(execucao,'ambiente',lambda root:environment)
    monkeypatch.setattr(correlacao.Steps,'guard',lambda self:None)
    config=copy.deepcopy(config)
    config['integridade']={'validar_entradas':True,'validar_saidas':True}
    data=root/'data';data.mkdir();days=['2025-01-01','2025-06-01']
    for i,day in enumerate(days):
        folder=data/f'date={day}';folder.mkdir()
        pq.write_table(pa.Table.from_pylist(raw_factory(day,start_block=21525891+i*1000)),folder/'part.parquet')
    tags=root/'tags.csv';shutil.copy2(RAIZ/'dados_auxiliares/labelcloud.csv',tags)
    paths={'dados':data,'labelcloud_csv':tags,'resultados':root/'resultados'}
    records=pipeline.selecionar(config,paths,'ensaio',days)
    ctx=execucao.Execucao(root,paths['resultados'],'original',config,paths,days,'ensaio')
    def save(path,value):escrever_json(path,value);return value
    with conectar(root/'tmp','512MB',1) as con:
        raw={};processed={}
        for r in records:
            output=ctx.path/'integridade-entradas'/f"{r['day']}.json"
            def verify(r=r,output=output):
                entries=[integridade.conferir(p) for p in r['files']]
                escrever_json(output,dict(day=r['day'],files=entries))
                return dict(rows=sum(e['rows'] for e in entries))
            ctx.etapa('verificar-entrada-'+r['day'],r['files'],[output],verify)
            output=ctx.path/'atributos'/r['window']/f"{r['day']}.parquet"
            ctx.etapa('atributos-'+r['day'],r['files'],[output],lambda r=r,p=output:atributos.gerar(con,r['files'],r['day'],p))
            raw.setdefault(r['window'],[]).append(output)
        output=ctx.path/'auditoria-global.json';all_raw=[p for files in raw.values() for p in files]
        ctx.etapa('auditoria-global',all_raw,[output],lambda:save(output,auditoria.auditar_global(con,all_raw)))
        scaler=ctx.path/'preprocessamento.json'
        ctx.etapa('ajustar-preprocessamento',raw['treino'],[scaler],
            lambda:save(scaler,preprocessamento.ajustar(con,raw['treino'],'train',ctx.contrato(raw['treino']))))
        params=ler_json(scaler)
        for r in records:
            source=raw[r['window']][0];output=ctx.path/'preprocessado'/r['window']/f"{r['day']}.parquet"
            ctx.etapa('transformar-'+r['day'],[source,scaler],[output],
                lambda s=source,p=output:preprocessamento.transformar(con,[s],params,p))
            processed.setdefault(r['window'],[]).append(output)
        output=ctx.path/'labelcloud.parquet'
        ctx.etapa('labelcloud-normalizar',[tags],[output],lambda:labelcloud.normalizar(tags,config['labelcloud'],output))
        output=ctx.path/'drift-B.json';all_processed=[p for files in processed.values() for p in files]
        ctx.etapa('drift-B',all_processed,[output],lambda:save(output,drift.analisar(con,processed,'treino',atributos.FEATURES,config['drift'])))
    ctx.manifest['scientific_status']='completed';ctx.finalizar('completed')
    rows=pq.ParquetFile(processed['treino'][0]).metadata.num_rows
    corr=correlacao.execute(ctx.path,ctx.manifest,processed['treino'],rows,ctx.path/'figuras','512MB',1,131072)
    ctx.manifest['correlation_outputs']=corr['outputs'];ctx.finalizar('completed');ctx.close()
    archive=tmp_path/'external/original';shutil.copytree(ctx.path,archive)
    new=copy.deepcopy(config);new['autoencoder'].update(learning_rate=.0003,weight_decay=0.)
    new['isolation_forest'].update(n_estimators=100,max_samples_tree=2048)
    return root,archive,new,paths,records


def plan_for(preparation):
    root,source,config,paths,records=preparation
    return reutilizacao.planejar(root,source,config,paths,'revision',records,'ensaio')


def test_importacao_preserva_origem_e_pipeline_pula_preparacao(preparation,monkeypatch):
    root,source,config,paths,records=preparation
    before={p:sha256(p) for p in source.rglob('*') if p.is_file()}
    plan=plan_for(preparation)
    assert not plan['target'].exists()
    assert not any(f['relative'].startswith(('modelos','escores','limiares','avaliacao'))
                   for stage in plan['stages'] for f in stage['files'])
    run=reutilizacao.executar(plan,config,paths,'revision','ensaio')
    manifest=ler_json(run/'manifesto.json')
    assert manifest['status']=='partial' and manifest['reuse']['status']=='completed'
    assert not manifest.get('scientific_status') and not (run/'modelos').exists()
    assert all(sha256(p)==digest for p,digest in before.items())
    for stage in plan['stages']:
        marker=ler_json(run/'.etapas'/f"{stage['key']}.json")
        assert marker['status']=='completed' and marker['reuse']['source']==str(source)
        assert all(sha256(run/f['relative'])==f['sha256'] for f in stage['files'])
    monkeypatch.setattr(pipeline,'RAIZ',root)
    def forbidden(*args,**kwargs):pytest.fail('A preparação não pode ser recalculada')
    for module,name in [(atributos,'gerar'),(preprocessamento,'ajustar'),(preprocessamento,'transformar'),
                        (auditoria,'auditar_global'),(integridade,'conferir')]:
        monkeypatch.setattr(module,name,forbidden)
    pipeline.executar(config,paths,'ensaio',[r['day'] for r in records],'revision','preprocessamento')
    from src import intersecao_mev,figura_intersecoes
    train_files=[run/f'preprocessado/treino/{r["day"]}.parquet' for r in records if r['window']=='treino']
    train_rows=sum(pq.ParquetFile(p).metadata.num_rows for p in train_files)
    monkeypatch.setattr(correlacao,'prepare',lambda path:(run,ler_json(run/'manifesto.json'),train_files,train_rows))
    monkeypatch.setattr(correlacao,'execute',forbidden)
    monkeypatch.setattr(intersecao_mev,'planejar',lambda *a:None)
    monkeypatch.setattr(intersecao_mev,'executar',lambda *a,**kw:dict(outputs={},status='completed'))
    monkeypatch.setattr(figura_intersecoes,'gerar',lambda *a:dict(outputs={},status='completed'))
    ctx=execucao.Execucao(root,paths['resultados'],'revision',config,paths,[r['day'] for r in records],'ensaio')
    try:
        ctx.etapa('labelcloud-normalizar',[paths['labelcloud_csv']],[run/'labelcloud.parquet'],forbidden)
        ctx.etapa('drift-B',[run/'preprocessado'/r['window']/f"{r['day']}.parquet" for r in records],
                  [run/'drift-B.json'],forbidden)
        pipeline.concluir(ctx)
        assert ctx.manifest['stages']['correlacoes']==dict(status='completed',reused=True)
    finally:ctx.close()


def test_copia_interrompida_retomavel(preparation,monkeypatch):
    _,_,config,paths,_=preparation;plan=plan_for(preparation)
    original=reutilizacao._copy_verified;calls=[]
    def fail(source,target,digest):
        calls.append(target)
        if len(calls)==3:raise RuntimeError('interrupção simulada')
        return original(source,target,digest)
    monkeypatch.setattr(reutilizacao,'_copy_verified',fail)
    with pytest.raises(RuntimeError,match='simulada'):
        reutilizacao.executar(plan,config,paths,'revision','ensaio')
    run=plan['target'];assert ler_json(run/'manifesto.json')['reuse']['status']=='failed'
    assert not (run/'.lock').exists() and (run/'.reuso/erros.log').is_file()
    monkeypatch.setattr(reutilizacao,'_copy_verified',original)
    reutilizacao.executar(plan_for(preparation),config,paths,'revision','ensaio')
    assert ler_json(run/'manifesto.json')['reuse']['status']=='completed'


def test_recusa_parquet_corrompido_e_bloqueia_pipeline(preparation,monkeypatch):
    root,source,config,paths,records=preparation;plan=plan_for(preparation)
    parquet=next(source.glob('atributos/*/*.parquet'))
    content=bytearray(parquet.read_bytes());content[20]^=1;parquet.write_bytes(content)
    with pytest.raises(ValueError,match='origem alterado'):
        reutilizacao.executar(plan,config,paths,'revision','ensaio')
    monkeypatch.setattr(pipeline,'RAIZ',root)
    with pytest.raises(ValueError,match='Importação incompleta'):
        pipeline.executar(config,paths,'ensaio',[r['day'] for r in records],'revision','preprocessamento')


def test_recusa_alteracao_de_codigo_ou_janelas(preparation):
    root,_,config,_,_=preparation
    config['janelas'][0]['fim_exclusivo']='2025-04-30'
    with pytest.raises(ValueError,match='janelas'):plan_for(preparation)
    config['janelas'][0]['fim_exclusivo']='2025-05-01'
    with (root/'src/atributos.py').open('a') as stream:stream.write('\n# altered\n')
    with pytest.raises(ValueError,match='Código incompatível'):plan_for(preparation)

import copy
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src import pipeline
from src.comum import ler_json,sha256,escrever_json,assinatura
from src.configuracao import RAIZ
from src.execucao import Execucao,codigo


def fake_root(tmp_path):
    root=tmp_path/'project';root.mkdir()
    for name in ('pyproject.toml','uv.lock'):
        shutil.copyfile(RAIZ/name,root/name)
    return root


def test_etapa_retomavel_recusa_entrada_ou_saida_alterada(tmp_path,config):
    root=fake_root(tmp_path)
    source=root/'input.txt';source.write_text('input')
    paths={'dados':root/'data','labelcloud_csv':root/'tags.csv','resultados':root/'results'}
    ctx=Execucao(root,paths['resultados'],'unit-test',config,paths,['2025-01-01'],'ensaio')
    output=ctx.path/'output.txt'
    called=[]
    def action():
        called.append(True);output.write_text('result');return {'rows':1}
    try:
        ctx.etapa('write',[source],[output],action)
        ctx.etapa('write',[source],[output],action)
        assert len(called)==1
        output.write_text('tampered')
        with pytest.raises(ValueError,match='Saída concluída alterada'):
            ctx.etapa('write',[source],[output],action)
        source.write_text('changed')
        with pytest.raises(ValueError,match='Entradas alteradas'):
            ctx.etapa('write',[source],[output],action)
        with pytest.raises(ValueError,match='fora da execução'):
            ctx.etapa('outside',[source],[root/'outside.txt'],action)
    finally:ctx.close()


def test_pipeline_sintetico_completo_e_retomada(tmp_path,raw_factory,config,monkeypatch):
    config['integridade']={'validar_entradas':True,'validar_saidas':True}
    root=fake_root(tmp_path)
    data=root/'data';data.mkdir()
    days=['2025-01-01','2025-06-01','2025-07-01']
    for n,day in enumerate(days):
        folder=data/f'date={day}';folder.mkdir()
        pq.write_table(pa.Table.from_pylist(raw_factory(day,start_block=21525891+n*1000)),folder/'part.parquet')
    label=root/'labelcloud.csv';shutil.copyfile(RAIZ/'dados_auxiliares/labelcloud.csv',label)
    paths={'dados':data,'labelcloud_csv':label,'resultados':root/'resultados'}
    monkeypatch.setattr(pipeline,'RAIZ',root)
    plan=pipeline.planejar(config,paths,'ensaio',days)
    assert plan['execution_started'] is False
    assert not paths['resultados'].exists()
    source_hashes={p:sha256(p) for p in data.rglob('*.parquet')}
    run=pipeline.executar(config,paths,'ensaio',days,'synthetic')
    first=ler_json(run/'manifesto.json')
    assert first['status']=='completed' and first['identity']['mode']=='ensaio'
    assert all(ler_json(run/'.etapas'/f'verificar-entrada-{day}.json')['result']['rows']>0 for day in days)
    assert ler_json(run/'.etapas/atributos-2025-01-01.json')['verification'][0]['rows']>0
    assert first['blockchain_coverage_verified'] is False
    assert all(sha256(p)==digest for p,digest in source_hashes.items())
    assert (run/'figuras/roc-pr.pdf').is_file()
    assert (run/'figuras/roc-pr-teste.pdf').is_file()
    assert (run/'figuras/matriz-teste-q995.pdf').is_file()
    assert (run/'figuras/correlacao-pearson-treino.pdf').is_file()
    assert (run/'figuras/correlacao-spearman-treino.pdf').is_file()
    assert (run/'figuras/intersecoes-mev-q995.pdf').is_file()
    assert (run/'figuras/intersecoes-mev-q995.png').is_file()
    intersections=ler_json(run/'avaliacao/intersecao-mev.json')
    assert set(intersections['windows'])=={'treino','validacao','teste'}
    assert all(str(run/relative) in first['artifacts'] for relative in (
        'avaliacao/intersecao-mev.json','figuras/intersecoes-mev-q995.pdf','figuras/intersecoes-mev-q995.png'))
    correlation=ler_json(run/'figuras/correlacao-treino.json')
    assert correlation['pearson']['rows']==ler_json(run/'.etapas/ajustar-if.json')['result']['training_rows_total']
    assert first['scientific_status']=='completed'
    assert len(list((run/'escores').rglob('*.parquet')))==len(days)*2
    assert ler_json(run/'limiares-AE.json')['role']=='validation'
    assert ler_json(run/'preprocessamento.json')['fit_role']=='train'
    for name in ('ajustar-ae','ajustar-if','ajustar-preprocessamento'):
        inputs=ler_json(run/'.etapas'/f'{name}.json')['inputs']
        assert not any('validacao/' in p.replace('\\','/') or 'teste/' in p.replace('\\','/') or 'labelcloud' in p for p in inputs)
    for method in ('AE','IF'):
        inputs=ler_json(run/'.etapas'/f'calibrar-{method}.json')['inputs']
        assert not any('/treino/' in p.replace('\\','/') or '/teste/' in p.replace('\\','/') or 'labelcloud' in p for p in inputs)
    assert ler_json(run/'avaliacao/teste.json')['models']['AE']['rows']==pq.ParquetFile(data/'date=2025-07-01/part.parquet').metadata.num_rows
    run2=pipeline.executar(config,paths,'ensaio',days,'synthetic')
    second=ler_json(run2/'manifesto.json')
    assert second['stages']==first['stages']
    assert second['artifacts']==first['artifacts']
    saved_concluir=pipeline.concluir
    def interrupted(ctx):
        raise RuntimeError('Falha simulada na conclusão')
    monkeypatch.setattr(pipeline,'concluir',interrupted)
    with pytest.raises(RuntimeError,match='Falha simulada'):
        pipeline.executar(config,paths,'ensaio',days,'synthetic')
    failed=ler_json(run/'manifesto.json')
    assert failed['status']=='failed' and failed['scientific_status']=='completed'
    monkeypatch.setattr(pipeline,'concluir',saved_concluir)
    monkeypatch.setattr(pipeline.atributos,'gerar',lambda *a:pytest.fail('Não pode repetir atributos'))
    pipeline.executar(config,paths,'ensaio',days,'synthetic')
    assert ler_json(run/'manifesto.json')['status']=='completed'
    bad=copy.deepcopy(config);bad['seed']=99
    with pytest.raises(ValueError,match='Configuração, código ou ambiente mudou'):
        pipeline.executar(bad,paths,'ensaio',days,'synthetic')
    assert not (run/'.lock').exists()


def test_continuidade_concluida_exige_identidade_exata(tmp_path,config):
    root=fake_root(tmp_path)
    paths={'dados':root/'data','labelcloud_csv':root/'tags.csv','resultados':root/'results'}
    ctx=Execucao(root,paths['resultados'],'completed-test',config,paths,['2025-01-01'],'ensaio')
    ctx.manifest['scientific_status']='completed';ctx.finalizar('completed');ctx.close()
    (root/'src').mkdir();(root/'src/add.py').write_text('# integrated reports')
    with pytest.raises(ValueError):Execucao(root,paths['resultados'],'completed-test',config,paths,['2025-01-01'],'ensaio')
    identity=copy.deepcopy(ctx.manifest['identity']);identity['code']=codigo(root)
    escrever_json(ctx.path/'continuidade.json',dict(source_signature=ctx.manifest['signature'],identity=identity,signature=assinatura(identity)))
    resumed=Execucao(root,paths['resultados'],'completed-test',config,paths,['2025-01-01'],'ensaio')
    resumed.close()
    changed=copy.deepcopy(config);changed['seed']=99
    with pytest.raises(ValueError):Execucao(root,paths['resultados'],'completed-test',changed,paths,['2025-01-01'],'ensaio')

import copy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src import intersecao_mev as mev,figura_intersecoes as figure
from src.comum import escrever_json,ler_json,sha256


def preparar(tmp_path):
    run=tmp_path/'resultados/unit';run.mkdir(parents=True)
    days=['2025-01-01','2025-01-02']
    inputs=[]
    for day in days:
        for relative,data in [
            (f'atributos/treino/{day}.parquet',{'row_id':[0,1,2],'from_address':['a','A','b']}),
            (f'escores/AE/treino/{day}.parquet',{'row_id':[0,1,2],'score':[.5,0.,1.]}),
            (f'escores/IF/treino/{day}.parquet',{'row_id':[0,1,2],'score':[.5,1.,1.]}),
        ]:
            target=run/relative;target.parent.mkdir(parents=True,exist_ok=True)
            pq.write_table(pa.table(data),target);inputs.append(target)
    tags=run/'labelcloud.parquet'
    pq.write_table(pa.table({'address':['A','a','b'],'is_mev_bot':[False,True,False]}),tags)
    inputs.append(tags)
    for method,count in [('AE',2),('IF',4)]:
        target=run/f'limiares-{method}.json'
        escrever_json(target,{'thresholds':{q:.5 for q in mev.QUANTILES}});inputs.append(target)
        target=run/f'labelcloud/treino-{method}.json'
        escrever_json(target,dict(rows=6,positives=4,thresholds={q:{'tp':count} for q in mev.QUANTILES}))
        inputs.append(target)
    escrever_json(run/'manifesto.json',dict(status='running',scientific_status='completed',signature='source',
        identity=dict(days=days,config={'janelas':[{'chave':'treino','inicio':'2025-01-01','fim_exclusivo':'2026-01-01'}]}),
        artifacts={str(p):sha256(p) for p in inputs}))
    return run


def test_interrupcao_reusa_dia_concluido_e_retomada_nao_reconta(tmp_path,monkeypatch):
    run=preparar(tmp_path);plan=mev.planejar(run,tmp_path)
    count=mev.contar_dia;calls=[]
    def fail_second(con,attributes,*args):
        calls.append(attributes.stem)
        if attributes.stem=='2025-01-02':raise RuntimeError('interrupção simulada')
        return count(con,attributes,*args)
    monkeypatch.setattr(mev,'contar_dia',fail_second)
    with pytest.raises(RuntimeError,match='simulada'):mev.executar(run,plan)
    first=run/'.intersecao-mev/dias/2025-01-01.json'
    first_hash=sha256(first)
    assert ler_json(run/'.intersecao-mev/manifesto.json')['status']=='failed'
    assert (run/'.intersecao-mev/erros.log').is_file()
    assert not (run/'.lock').exists()
    calls.clear()
    def resume(con,attributes,*args):
        calls.append(attributes.stem);return count(con,attributes,*args)
    monkeypatch.setattr(mev,'contar_dia',resume)
    with mev.lock_run(run):
        result=mev.executar(run,plan,already_locked=True)
        assert (run/'.lock').exists()
    assert calls==['2025-01-02'] and sha256(first)==first_hash
    output=ler_json(run/'avaliacao/intersecao-mev.json')['windows']['treino']
    assert output['thresholds']['q995']==dict(AE=2,IF=4,intersection=2,union=4,AE_only=0,IF_only=2,neither=0)
    monkeypatch.setattr(mev,'contar_dia',lambda *args:pytest.fail('Não pode recontar'))
    assert mev.executar(run,plan)==result
    (run/'escores/AE/treino/2025-01-01.parquet').write_bytes(b'corrompido')
    with pytest.raises(ValueError,match='Entrada alterada'):mev.executar(run,plan)


def test_continuidade_de_intersecoes_exige_identidade_exata(tmp_path):
    identity={'inputs':{},'code_sha256':'new','packages':{}}
    prior_identity={**identity,'code_sha256':'old'}
    prior={'identity':prior_identity,'signature':mev.signature(prior_identity)}
    path=tmp_path/'continuidade.json'
    with pytest.raises(ValueError,match='mudaram'):mev.identidade_retomada(identity,prior,path)
    escrever_json(path,dict(source_signature=prior['signature'],identity=identity,signature=mev.signature(identity)))
    assert mev.identidade_retomada(identity,prior,path)==prior_identity
    with pytest.raises(ValueError,match='mudaram'):
        mev.identidade_retomada({**identity,'inputs':{'changed':'hash'}},prior,path)


def test_bloqueio_do_pipeline_nao_pode_ser_assumido_sem_posse(tmp_path):
    with pytest.raises(RuntimeError,match='não possui'):
        with mev.lock_run(tmp_path,already_locked=True):pass


def test_figura_retomavel_aceita_uniao_vazia_e_recusa_saida_alterada(tmp_path,monkeypatch):
    run=preparar(tmp_path);mev.executar(run,mev.planejar(run,tmp_path))
    document=ler_json(run/'avaliacao/intersecao-mev.json')
    empty=copy.deepcopy(document['windows']['treino'])
    empty['thresholds']['q995']={k:0 for k in empty['thresholds']['q995']}
    document['windows']['teste']=empty
    escrever_json(run/'avaliacao/intersecao-mev.json',document)
    first=figure.gerar(run)
    assert all(Path(p).is_file() for p in first['outputs'])
    monkeypatch.setattr(figure,'plot',lambda *a:pytest.fail('Não pode redesenhar'))
    assert figure.gerar(run)==first
    (run/'figuras/intersecoes-mev-q995.png').write_bytes(b'alterado')
    with pytest.raises(ValueError,match='Checkpoint alterado'):figure.gerar(run)


def test_reconciliacao_recusa_particao_inconsistente():
    row=dict(AE=2,IF=2,intersection=1,union=3,AE_only=1,IF_only=1,neither=1)
    actual={'associated':4,'thresholds':{q:copy.deepcopy(row) for q in mev.QUANTILES}}
    reports={'positives':4,'thresholds':{q:{'tp':2} for q in mev.QUANTILES}}
    actual['thresholds']['q995']['AE_only']=2
    with pytest.raises(ValueError,match='inconsistente'):mev.reconciliar(actual,reports,reports)

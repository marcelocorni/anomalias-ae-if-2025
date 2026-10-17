import os
import sys
from types import SimpleNamespace

import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src import operacao
from src.comum import escrever_json,ler_json
from src.execucao import Execucao
from testes.test_execucao import fake_root


def test_inventario_recusa_populacao_diferente(tmp_path):
    p=tmp_path/'input.parquet';pq.write_table(pa.table({'x':[1,2,3]}),p)
    assert operacao.conferir_arquivos([p],1,3)['rows']==3
    with pytest.raises(ValueError):operacao.conferir_arquivos([p],1,4)
    with pytest.raises(ValueError):operacao.conferir_arquivos([p],2,3)


def test_lock_nao_confunde_pid_reutilizado(tmp_path):
    lock=tmp_path/'lock.json'
    created=psutil.Process().create_time()
    escrever_json(lock,dict(pid=os.getpid(),created=created))
    assert operacao.processo_ativo(lock)
    escrever_json(lock,dict(pid=os.getpid(),created=created-100))
    assert not operacao.processo_ativo(lock)


def test_falha_tem_log_codigo_e_nao_reexecuta(tmp_path,monkeypatch):
    monkeypatch.setattr(operacao.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1))
    operations=tmp_path/'operations';run=tmp_path/'results'
    command=[sys.executable,'-u','-c',"import sys,time;print('etapa-sintetica');print('erro-sintetico',file=sys.stderr);time.sleep(.2);sys.exit(7)"]
    assert operacao.monitorar(command,tmp_path,run,operations,.05,False)==7
    state=ler_json(operations/'estado.json')
    assert state['status']=='failed' and state['exit_code']==7
    log=open(state['log'],encoding='utf-8').read()
    assert log.count('etapa-sintetica')==1 and 'erro-sintetico' in log
    assert open(state['resources'],encoding='utf-8').read().strip()
    assert not list(operations.glob('*-monitor-error.log'))
    assert not (operations/'.supervisor.lock').exists()


def test_preflight_planeja_sem_iniciar_e_confere_cuda(tmp_path,config,monkeypatch):
    from scripts import executar_etapa4
    root=tmp_path/'project';(root/'configuracao').mkdir(parents=True)
    source=root/'input.parquet';pq.write_table(pa.table({'x':[1,2,3]}),source)
    policy=dict(id='anual-test',config='configuracao/etapa4.json',local='configuracao/local.json',expected_files=1,expected_rows=3,initial_free_gib=0)
    escrever_json(root/'configuracao/execucao_etapa4.json',policy)
    escrever_json(root/'configuracao/etapa4.json',config)
    paths={'dados':root/'data','labelcloud_csv':root/'tags.csv','resultados':root/'results'}
    config['autoencoder']['device']='cuda'
    monkeypatch.setattr(executar_etapa4,'RAIZ',root)
    monkeypatch.setattr(executar_etapa4,'carregar',lambda *a:(config,paths))
    monkeypatch.setattr(executar_etapa4,'selecionar',lambda *a:[dict(files=[source])])
    monkeypatch.setattr(executar_etapa4.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='{"available":true,"cuda":"13.0","gpu":"fixture"}'))
    _,_,run,command,report=executar_etapa4.preparar()
    assert not run.exists() and report['annual_execution_started'] is False
    assert report['inventory']['rows']==3 and report['mode']=='completo'
    assert '--dias' not in command and command[-1]=='--executar'


def test_sucesso_e_nova_tentativa_preservam_logs(tmp_path,monkeypatch):
    monkeypatch.setattr(operacao.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1))
    operations=tmp_path/'operations';run=tmp_path/'results'
    command=[sys.executable,'-c',"from pathlib import Path;import json;p=Path('results');p.mkdir(exist_ok=True);(p/'manifesto.json').write_text(json.dumps({'status':'completed'}));print('concluido')"]
    assert operacao.monitorar(command,tmp_path,run,operations,.05,False)==0
    first=ler_json(operations/'estado.json')
    assert first['status']=='completed'
    assert operacao.monitorar(command,tmp_path,run,operations,.05,False)==0
    second=ler_json(operations/'estado.json')
    assert first['log']!=second['log']
    assert len(list(operations.glob('*-estado.json')))==2
    assert not (operations/'.supervisor.lock').exists()


def test_retomada_apos_lock_obsoleto_preserva_etapa(tmp_path,config):
    root=fake_root(tmp_path)
    paths={'dados':root/'data','labelcloud_csv':root/'tags.csv','resultados':root/'results'}
    source=root/'input.txt';source.write_text('input')
    ctx=Execucao(root,paths['resultados'],'reboot-test',config,paths,['2025-01-01'],'ensaio')
    output=ctx.path/'output.txt'
    def action():output.write_text('done');return {'rows':1}
    ctx.etapa('one',[source],[output],action)
    ctx.close()
    escrever_json(ctx.path/'.lock',dict(pid=os.getpid(),created=psutil.Process().create_time()-100))
    resumed=Execucao(root,paths['resultados'],'reboot-test',config,paths,['2025-01-01'],'ensaio')
    try:
        result=resumed.etapa('one',[source],[output],lambda:pytest.fail('Etapa concluída não pode ser reexecutada.'))
        assert result['rows']==1 and output.read_text()=='done'
    finally:resumed.close()

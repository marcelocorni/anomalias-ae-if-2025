import copy
from types import SimpleNamespace

import pytest

from scripts import executar_etapa4 as launcher
from src import configuracao
from src.comum import ler_json,escrever_json,sha256


def test_perfil_utiliza_vencedores_sem_alterar_outros_parametros():
    root=configuracao.RAIZ
    original=ler_json(root/'configuracao/experimento.json')
    revision=ler_json(root/'configuracao/experimento-revisao-01.json')
    expected=copy.deepcopy(original)
    expected['experimento']='sbrc2027-b-2025-revisao-01'
    expected['autoencoder'].update(learning_rate=.0003,weight_decay=0.)
    expected['isolation_forest'].update(n_estimators=100,max_samples_tree=2048)
    expected['selecao_hiperparametros']=revision['selecao_hiperparametros']
    assert revision==expected
    configuracao.validar_hiperparametros_fixos(revision)
    profile=ler_json(root/'configuracao/execucao_revisao_01.json')
    assert profile['id']=='anual-2025-b-revisao-01'
    assert profile['config']=='configuracao/experimento-revisao-01.json'
    assert profile['initial_free_gib']==200
    assert revision['selecao_hiperparametros']['sha256']=='c443031edbfd7990f8adeebfbda2b9ddd5954f548b0548ac1b736ffaa90e1086'


@pytest.mark.parametrize('section,key,value',[
    ('autoencoder','learning_rate',.001),('autoencoder','weight_decay',1e-5),
    ('isolation_forest','n_estimators',200),('isolation_forest','max_samples_tree',8192),
])
def test_recusa_parametros_diferentes_dos_vencedores(section,key,value):
    config=ler_json(configuracao.RAIZ/'configuracao/experimento-revisao-01.json')
    config[section][key]=value
    with pytest.raises(ValueError,match='divergem'):configuracao.validar_hiperparametros_fixos(config)


def test_recusa_referencia_alterada(tmp_path,monkeypatch):
    original_root=configuracao.RAIZ
    config=ler_json(original_root/'configuracao/experimento-revisao-01.json')
    target=tmp_path/config['selecao_hiperparametros']['referencia']
    target.parent.mkdir()
    target.write_bytes((original_root/config['selecao_hiperparametros']['referencia']).read_bytes()+b' ')
    monkeypatch.setattr(configuracao,'RAIZ',tmp_path)
    with pytest.raises(ValueError,match='Hash'):configuracao.validar_hiperparametros_fixos(config)


def test_preflight_usa_perfil_explicito_sem_iniciar_pipeline(tmp_path,monkeypatch):
    root=configuracao.RAIZ
    config=ler_json(root/'configuracao/experimento-revisao-01.json')
    profile=ler_json(root/'configuracao/execucao_revisao_01.json')
    profile.update(expected_files=0,expected_rows=0,initial_free_gib=0,reuse_required=False)
    (tmp_path/'configuracao').mkdir()
    escrever_json(tmp_path/'configuracao/perfil.json',profile)
    escrever_json(tmp_path/profile['config'],config)
    config['recursos']['min_free_gib']=0
    paths={'dados':tmp_path/'dados','labelcloud_csv':tmp_path/'tags.csv','resultados':tmp_path/'resultados'}
    monkeypatch.setattr(launcher,'RAIZ',tmp_path)
    monkeypatch.setattr(launcher,'carregar',lambda *a:(config,paths))
    monkeypatch.setattr(launcher,'selecionar',lambda *a:[])
    monkeypatch.setattr(launcher,'conferir_arquivos',lambda *a:{'files':0,'rows':0})
    monkeypatch.setattr(launcher.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='{"available":true,"cuda":"13.0","gpu":"fixture"}'))
    monkeypatch.setattr(launcher,'monitorar',lambda *a:pytest.fail('Não pode executar o pipeline'))
    _,_,run,command,report=launcher.preparar('configuracao/perfil.json')
    assert run==paths['resultados']/'anual-2025-b-revisao-01'
    assert not run.exists() and report['annual_execution_started'] is False
    assert report['hyperparameters']['AE']['learning_rate']==.0003
    assert report['hyperparameters']['IF']['n_estimators']==100
    assert command[command.index('--config')+1]==str(tmp_path/profile['config'])


def test_perfil_bloqueia_pipeline_antes_da_importacao(tmp_path,monkeypatch):
    config=ler_json(configuracao.RAIZ/'configuracao/experimento-revisao-01.json')
    profile=ler_json(configuracao.RAIZ/'configuracao/execucao_revisao_01.json')
    (tmp_path/'configuracao').mkdir()
    escrever_json(tmp_path/'configuracao/perfil.json',profile)
    paths={'dados':tmp_path/'dados','labelcloud_csv':tmp_path/'tags.csv','resultados':tmp_path/'resultados'}
    monkeypatch.setattr(launcher,'RAIZ',tmp_path)
    monkeypatch.setattr(launcher,'carregar',lambda *a:(config,paths))
    monkeypatch.setattr(launcher,'selecionar',lambda *a:[])
    monkeypatch.setattr(launcher,'conferir_arquivos',lambda *a:{'files':0,'rows':0})
    monkeypatch.setattr(launcher.subprocess,'run',lambda *a,**k:pytest.fail('Não pode iniciar nenhum processo'))
    with pytest.raises(ValueError,match='Importe e confira'):launcher.preparar('configuracao/perfil.json')
    assert not paths['resultados'].exists()

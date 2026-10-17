import pytest

from src import configuracao
from src.comum import escrever_json


def preparar_local(tmp_path,config,monkeypatch,output):
    root=tmp_path/'project';root.mkdir()
    data=tmp_path/'source';data.mkdir()
    label=root/'dados_auxiliares/labelcloud.csv';label.parent.mkdir();label.write_text('fixture')
    cfg=root/'experimento.json';local=root/'local.json'
    escrever_json(cfg,config)
    paths={'dados':str(data),'labelcloud_csv':str(label),'resultados':str(output(root,data))}
    escrever_json(local,paths)
    monkeypatch.setattr(configuracao,'RAIZ',root)
    return cfg,local


@pytest.mark.parametrize('destination',[lambda r,d:d/'derived',lambda r,d:r.parent/'outside',lambda r,d:r/'artefatos/results',lambda r,d:r/'src/results',lambda r,d:r])
def test_configuracao_protege_dados_e_originais(tmp_path,config,monkeypatch,destination):
    cfg,local=preparar_local(tmp_path,config,monkeypatch,destination)
    with pytest.raises(ValueError):configuracao.carregar(cfg,local)


def test_configuracao_nao_admite_amostragem_de_blocos(tmp_path,config,monkeypatch):
    config['percentual_blocos']=1
    cfg,local=preparar_local(tmp_path,config,monkeypatch,lambda r,d:r/'resultados')
    with pytest.raises(ValueError,match='100%'):configuracao.carregar(cfg,local)

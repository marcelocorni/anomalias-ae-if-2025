from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from src import autoencoder,isolation_forest
from src.atributos import gerar
from src.dados import gravar_escores
from src.preprocessamento import ajustar,transformar


def preparar(con,write_raw,tmp_path):
    raw=write_raw();features=tmp_path/'B.parquet';processed=tmp_path/'preprocessed.parquet'
    gerar(con,[raw],'2025-01-01',features)
    params=ajustar(con,[features],'train','signature')
    transformar(con,[features],params,processed)
    return processed


def test_inferencia_ae_if_independe_do_tamanho_do_lote(con,write_raw,tmp_path,config):
    train=preparar(con,write_raw,tmp_path)
    isolation_forest.treinar(con,[train],config['isolation_forest'],42,1,tmp_path/'if.joblib','contract')
    autoencoder.treinar(con,[train],config['autoencoder'],42,tmp_path/'ae','contract',tmp_path/'scratch')
    scorers={'ae':autoencoder.carregar(tmp_path/'ae/modelo.pt','cpu'),'if':isolation_forest.carregar(tmp_path/'if.joblib',1)}
    for name,scorer in scorers.items():
        targets=[tmp_path/f'{name}-{batch}.parquet' for batch in (2,17)]
        for batch,path in zip((2,17),targets):gravar_escores([train],path,scorer,batch)
        left,right=[pq.read_table(p) for p in targets]
        assert left['row_id'].to_pylist()==right['row_id'].to_pylist()
        np.testing.assert_allclose(left['score'].to_numpy(),right['score'].to_numpy(),rtol=1e-6,atol=1e-7)


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA indisponível'))])
def test_checkpoint_ae_reproduz_execucao_sem_interrupcao(con,write_raw,tmp_path,config,monkeypatch,device):
    train=preparar(con,write_raw,tmp_path)
    config['autoencoder'].update(device=device,amp=device=='cuda')
    config['autoencoder']['max_epochs']=3
    autoencoder.treinar(con,[train],config['autoencoder'],42,tmp_path/'continuous','contract',tmp_path/'scratch1')
    real_save=autoencoder.salvar_torch
    raised=False
    def interrupted(path,data):
        nonlocal raised
        real_save(path,data)
        if Path(path).name=='checkpoint.pt' and not raised:
            raised=True
            raise RuntimeError('Interrupção sintética após checkpoint completo.')
    monkeypatch.setattr(autoencoder,'salvar_torch',interrupted)
    with pytest.raises(RuntimeError,match='Interrupção sintética'):
        autoencoder.treinar(con,[train],config['autoencoder'],42,tmp_path/'resumed','contract',tmp_path/'scratch2')
    monkeypatch.setattr(autoencoder,'salvar_torch',real_save)
    autoencoder.treinar(con,[train],config['autoencoder'],42,tmp_path/'resumed','contract',tmp_path/'scratch2')
    a=torch.load(tmp_path/'continuous/modelo.pt',weights_only=False)
    b=torch.load(tmp_path/'resumed/modelo.pt',weights_only=False)
    assert all(torch.equal(a['state_dict'][key],b['state_dict'][key]) for key in a['state_dict'])
    with pytest.raises(ValueError,match='outro contrato'):
        autoencoder.treinar(con,[train],config['autoencoder'],42,tmp_path/'resumed','changed',tmp_path/'scratch2')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA indisponível')
def test_cuda_amp_em_um_lote_sintetico():
    device=autoencoder.configurar(42,'cuda')
    model=autoencoder.Autoencoder().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    scaler=torch.amp.GradScaler('cuda')
    values=torch.linspace(-1,1,32*21,device=device).reshape(32,21)
    with torch.amp.autocast(device_type='cuda',dtype=torch.float16):
        loss=torch.nn.functional.mse_loss(model(values),values)
    assert torch.isfinite(loss)
    scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
    with torch.inference_mode():
        scores=torch.mean((model(values).float()-values.float())**2,dim=1)
    assert scores.shape==(32,) and torch.isfinite(scores).all()

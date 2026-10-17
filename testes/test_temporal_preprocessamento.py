import copy

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.atributos import FEATURES, gerar
from src.calibracao import calibrar
from src.comum import assinatura
from src.preprocessamento import ROBUST, ajustar, transformar
from src.recortes import dias_do_ano, janela_do_dia, validar_janelas


def test_janelas_cobrem_365_dias_sem_vazamento(config):
    validar_janelas(config['janelas'])
    assert len(dias_do_ano())==365
    assert janela_do_dia('2025-04-30',config['janelas'])['papel']=='train'
    assert janela_do_dia('2025-05-01',config['janelas'])['papel']=='retrospective'
    assert janela_do_dia('2025-06-01',config['janelas'])['papel']=='validation'
    assert janela_do_dia('2025-07-01',config['janelas'])['papel']=='test'
    assert janela_do_dia('2025-12-10',config['janelas'])['papel']=='future_stress'
    with pytest.raises(ValueError):janela_do_dia('2026-01-01',config['janelas'])
    bad=copy.deepcopy(config['janelas']);bad[1]['inicio']='2025-04-30'
    with pytest.raises(ValueError):validar_janelas(bad)


def test_scaler_somente_treino_condicional_e_congelado(con,write_raw,tmp_path):
    raw=write_raw();train=tmp_path/'train.parquet'
    gerar(con,[raw],'2025-01-01',train)
    before=ajustar(con,[train],'train','input')
    signature=before['signature']
    table=pq.read_table(train)
    original=table.to_pylist()
    for feature in ROBUST:
        applicable=[r[feature] for r in original if feature not in ('max_fee_per_gas_log','max_priority_fee_log') or r['tx_type_0']==r['tx_type_1']==0]
        assert before['parameters'][feature]['median']==pytest.approx(float(np.median(applicable)))
    extreme=[dict(r,**{f:r[f]*10000 for f in ROBUST}) for r in original]
    val=tmp_path/'validation.parquet';pq.write_table(pa.Table.from_pylist(extreme),val)
    transformed=tmp_path/'transformed.parquet'
    transformar(con,[val],before,transformed)
    assert before['signature']==signature
    assert set(pq.read_schema(transformed).names)=={'row_id',*FEATURES}
    for a,b in zip(extreme,pq.read_table(transformed).to_pylist()):
        if a['tx_type_0'] or a['tx_type_1']:
            assert b['max_fee_per_gas_log']==b['max_priority_fee_log']==0
    for role in ('validation','test'):
        with pytest.raises(ValueError):ajustar(con,[val],role,'input')
    corrupt=copy.deepcopy(before);corrupt['parameters']['nonce_log']['median']=999
    with pytest.raises(ValueError):transformar(con,[val],corrupt,tmp_path/'corrupt.parquet')


def test_iqr_zero_usa_escala_unitaria(con,write_raw,tmp_path):
    raw=write_raw();train=tmp_path/'B.parquet';gerar(con,[raw],'2025-01-01',train)
    rows=pq.read_table(train).to_pylist()
    for row in rows:row['nonce_log']=3.
    pq.write_table(pa.Table.from_pylist(rows),train)
    document=ajustar(con,[train],'train','input')
    assert document['parameters']['nonce_log']['scale']==1.
    target=tmp_path/'scaled.parquet';transformar(con,[train],document,target)
    assert set(pq.read_table(target)['nonce_log'].to_pylist())=={0.}


def test_calibracao_exata_so_na_validacao(con,tmp_path,config):
    values=np.array([0,0,1,1,1,2,20,100],dtype=np.float32)
    p=tmp_path/'scores.parquet';pq.write_table(pa.table({'row_id':np.arange(len(values),dtype=np.uint64),'score':values}),p)
    result=calibrar(con,[p],'validation',config['calibracao'],'model','input')
    for key,quantile in config['calibracao'].items():
        assert result['thresholds'][key]==pytest.approx(float(np.quantile(values,quantile)),rel=1e-6)
    assert result['signature']==assinatura({k:v for k,v in result.items() if k!='signature'})
    for role in ('train','test'):
        with pytest.raises(ValueError):calibrar(con,[p],role,config['calibracao'],'model','input')

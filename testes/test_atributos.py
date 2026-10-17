import numpy as np
import pyarrow.parquet as pq
import pytest

from src.atributos import FEATURES, gerar
from src.comum import parquet
from testes.fixtures.legado_atributos import create_feature_master as legacy


def test_equivalencia_dos_21_atributos_com_legado(con,write_raw,tmp_path):
    source=write_raw()
    con.execute('CREATE VIEW raw_transactions AS SELECT * FROM '+parquet([source]))
    legacy(con,1)
    original=con.execute('SELECT '+','.join(FEATURES)+' FROM feature_master ORDER BY hash').fetchnumpy()
    output=tmp_path/'B.parquet'
    audit=gerar(con,[source],'2025-01-01',output)
    migrated=con.execute('SELECT '+','.join(FEATURES)+' FROM '+parquet([output])+' ORDER BY hash').fetchnumpy()
    assert len(FEATURES)==21
    for feature in FEATURES:
        np.testing.assert_allclose(migrated[feature],original[feature],rtol=1e-13,atol=1e-13)
    assert audit['rows']==pq.ParquetFile(source).metadata.num_rows
    assert audit['blockchain_coverage_verified'] is False


def test_blocos_divididos_em_arquivos_mantem_contexto_e_ids(con,raw_factory,write_raw,tmp_path):
    rows=raw_factory()
    full=write_raw(rows=rows,name='full.parquet')
    one=write_raw(rows=rows[::2],name='part1.parquet')
    two=write_raw(rows=rows[1::2],name='part2.parquet')
    a,b=tmp_path/'a.parquet',tmp_path/'b.parquet'
    gerar(con,[full],'2025-01-01',a)
    gerar(con,[two,one],'2025-01-01',b)
    assert pq.read_table(a).equals(pq.read_table(b))
    values=pq.read_table(a).to_pylist()
    assert values[0]['row_id']==(values[0]['block_number']<<32)|values[0]['transaction_index']
    single=next(r for r in values if r['block_number']==21525891)
    assert single['transaction_index_pct']==0 and single['gas_price_rank_block']==0


@pytest.mark.parametrize('case',['duplicata','indice_ausente','data_errada','nulo','coluna_ausente','infinito'])
def test_integridade_recusa_dados_invalidos(case,con,raw_factory,write_raw,tmp_path):
    rows=raw_factory()
    if case=='duplicata':rows.append(rows[0].copy())
    elif case=='indice_ausente':rows[1]['transaction_index']=4
    elif case=='data_errada':rows[0]['date']='2025-01-02'
    elif case=='nulo':rows[0]['receipt_gas_used']=None
    elif case=='infinito':rows[0]['value']=float('inf')
    else:
        for row in rows:row.pop('nonce')
    source=write_raw(rows=rows)
    with pytest.raises((ValueError,RuntimeError)):
        gerar(con,[source],'2025-01-01',tmp_path/'bad.parquet')


def test_fallback_zero_nao_e_substituido_por_gas_price(con,raw_factory,write_raw,tmp_path):
    rows=raw_factory()
    rows[0]['receipt_effective_gas_price']=0
    source=write_raw(rows=rows)
    output=tmp_path/'out.parquet'
    gerar(con,[source],'2025-01-01',output)
    result=pq.read_table(output).to_pylist()
    assert next(r for r in result if r['hash']==rows[0]['hash'])['effective_gas_price_log']==0

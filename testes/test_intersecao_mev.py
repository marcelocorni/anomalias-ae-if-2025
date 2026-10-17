import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.intersecao_mev import contar_dia,reconciliar,lock_run,QUANTILES
from src.comum import conectar


@pytest.mark.parametrize('addresses,flags',[
    (['a','b'],[True,False]),
    (['A','a','a','b','B'],[False,True,True,False,False]),
])
def test_intersecao_restringe_associacao_e_inclui_empates(tmp_path,addresses,flags):
    tags=tmp_path/'tags.parquet';raw=tmp_path/'raw.parquet';ae=tmp_path/'ae.parquet';iff=tmp_path/'if.parquet'
    pq.write_table(pa.table({'address':addresses,'is_mev_bot':flags}),tags)
    pq.write_table(pa.table({'row_id':list(range(6)),'from_address':['a','a','a','a','b','c']}),raw)
    pq.write_table(pa.table({'row_id':list(range(6)),'score':[1.,.5,0.,0.,1.,1.]}),ae)
    pq.write_table(pa.table({'row_id':list(range(6)),'score':[.5,0.,1.,0.,1.,1.]}),iff)
    thresholds={m:{q:.5 for q in QUANTILES} for m in ('AE','IF')}
    with conectar(tmp_path/'duckdb','128MB',1) as con:
        result=contar_dia(con,raw,ae,iff,tags,thresholds)
    assert result['associated']==4
    assert result['thresholds']['q995']==dict(AE=2,IF=2,intersection=1,union=3,AE_only=1,IF_only=1,neither=1)
    expected=dict(positives=4,thresholds={q:{'tp':2} for q in QUANTILES})
    reconciliar(result,expected,expected)
    expected['thresholds']['q995']['tp']=3
    with pytest.raises(ValueError,match='divergem'):reconciliar(result,expected,expected)
    pq.write_table(pa.table({'row_id':list(range(1,6)),'score':[.5,0.,0.,1.,1.]}),ae)
    with conectar(tmp_path/'duckdb-missing','128MB',1) as con:
        with pytest.raises(ValueError,match='Alinhamento'):
            contar_dia(con,raw,ae,iff,tags,thresholds)


def test_lock_impede_execucoes_simultaneas_e_libera(tmp_path):
    with lock_run(tmp_path):
        with pytest.raises(RuntimeError,match='ativa'):
            with lock_run(tmp_path):pass
    assert not (tmp_path/'.lock').exists()

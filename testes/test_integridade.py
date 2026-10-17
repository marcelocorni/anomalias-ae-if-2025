import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src import integridade
from src.comum import ler_json
from src.execucao import Execucao
from testes.test_execucao import fake_root


def test_conferencia_detecta_leitura_inconsistente(tmp_path, monkeypatch):
    p=tmp_path/'p.parquet';pq.write_table(pa.table({'a':[1,2]}),p)
    assert integridade.conferir(p)['rows']==2
    values=iter(['a','b'])
    monkeypatch.setattr(integridade,'sha256',lambda p:next(values))
    with pytest.raises(ValueError,match='inconsistentes'):integridade.conferir(p)


def test_comparacao_preserva_antigo_e_separa_ausencia(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    (old/'a.json').write_text('old');(new/'a.json').write_text('new')
    result=integridade.comparar(new/'a.json',new,['old','missing'])
    assert [r['status'] for r in result]==['different_bytes','absent']
    assert (old/'a.json').read_text()=='old'


def test_saida_corrompida_nao_conclui_etapa(tmp_path,config):
    root=fake_root(tmp_path)
    config['integridade']={'validar_saidas':True}
    source=root/'input';source.write_text('data')
    ctx=Execucao(root,root/'results','checked',config,{},['2025-01-01'],'ensaio')
    output=ctx.path/'bad.parquet'
    def action():output.write_bytes(b'invalid parquet');return {}
    try:
        with pytest.raises(Exception):ctx.etapa('bad',[source],[output],action)
        assert ler_json(ctx.path/'.etapas/bad.json')['status']=='failed'
        assert output.read_bytes()==b'invalid parquet'
    finally:ctx.close()


def test_comparacao_distingue_codificacao_de_conteudo(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    table=pa.table({'x':[1,2,3]})
    pq.write_table(table,old/'a.parquet',compression='NONE')
    pq.write_table(table,new/'a.parquet',compression='zstd')
    result=integridade.comparar(new/'a.parquet',new,['old'])[0]
    assert result['status']=='different_bytes'
    assert result['regenerated_minus_previous']==result['previous_minus_regenerated']==0
    pq.write_table(pa.table({'x':[1,2,4]}),new/'a.parquet')
    result=integridade.comparar(new/'a.parquet',new,['old'])[0]
    assert result['regenerated_minus_previous']==result['previous_minus_regenerated']==1


def test_antigo_ilegivel_e_preservado_sem_invalidar_saida_nova(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    invalid=pa.array([b'\xff']).cast(pa.string(),safe=False)
    pq.write_table(pa.table({'text':invalid}),old/'a.parquet')
    pq.write_table(pa.table({'text':['valid']}),new/'a.parquet')
    previous=(old/'a.parquet').read_bytes()
    result=integridade.comparar(new/'a.parquet',new,['old'])[0]
    assert result['status']=='previous_unreadable' and result['error']
    assert (old/'a.parquet').read_bytes()==previous
    assert integridade.conferir(new/'a.parquet')['rows']==1

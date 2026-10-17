import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.stats import spearmanr

from src import correlacao as correlation
from src.comum import conectar


def test_pearson_em_lotes_e_constante():
    x=np.array([[1.,4.,1.23],[2.,1.,1.23],[2.,3.,1.23],[4.,2.,1.23]])
    state=correlation.merge(correlation.moments(x[:2]),correlation.moments(x[2:]))
    r,constant=correlation.coefficients(state)
    np.testing.assert_allclose(r[:2,:2],np.corrcoef(x[:,:2],rowvar=False),atol=1e-12)
    assert constant.tolist()==[False,False,True]
    assert np.isnan(r[2,:]).all() and np.isnan(r[:,2]).all()


def test_spearman_exato_com_empates(tmp_path):
    source=tmp_path/'input.parquet';source_table=pa.table({'x':[1.,2.,2.,4.],'y':[4.,1.,3.,2.]})
    pq.write_table(source_table,source)
    maps=[]
    try:
        with conectar(tmp_path/'duckdb','128MB',1) as con:
            for name in ('x','y'):
                target=tmp_path/(name+'.parquet')
                correlation.map_ranks(con,[source],name,target)
                maps.append((np.load(target.with_suffix('.values.npy'),mmap_mode='r'),np.load(target.with_suffix('.ranks.npy'),mmap_mode='r')))
        values=np.array([[1,4],[2,1],[2,3],[4,2]])
        ranks=correlation.rank_values(values,maps)
        np.testing.assert_array_equal(ranks[:,0],[1,2.5,2.5,4])
        r,_=correlation.coefficients(correlation.moments(ranks))
        assert r[0,1]==pytest.approx(spearmanr(values[:,0],values[:,1]).statistic,abs=1e-12)
    finally:
        for values,ranks in maps:values._mmap.close();ranks._mmap.close()


def test_checkpoint_nao_reexecuta_e_recusa_saida_alterada(tmp_path):
    steps=correlation.Steps(tmp_path,'contract',reserve_gib=0)
    target=tmp_path/'saved.txt';calls=[]
    def action():calls.append(1);target.write_text('value');return {'done':True}
    steps.run('one',[target],action);steps.run('one',[target],action)
    assert len(calls)==1
    target.write_text('tampered')
    with pytest.raises(ValueError,match='Checkpoint alterado'):steps.run('one',[target],action)


def test_postos_recusam_valor_ausente():
    with pytest.raises(ValueError,match='ausente'):
        correlation.rank_values(np.array([[3]]),[(np.array([1.,2.]),np.array([1.,2.]))])


def test_gravacao_repete_somente_bloqueio_transitorio(tmp_path,monkeypatch):
    from pathlib import Path
    original=Path.replace;calls=[]
    def blocked(source,target):
        calls.append(1)
        if len(calls)<3:
            exc=PermissionError('locked');exc.winerror=5;raise exc
        return original(source,target)
    monkeypatch.setattr(Path,'replace',blocked)
    monkeypatch.setattr(correlation.time,'sleep',lambda _:None)
    target=tmp_path/'marker.json'
    correlation.write(target,{'done':True})
    assert len(calls)==3 and correlation.read(target)=={'done':True}
    calls.clear()
    def always_blocked(source,target):
        calls.append(1);exc=PermissionError('locked');exc.winerror=5;raise exc
    monkeypatch.setattr(Path,'replace',always_blocked)
    with pytest.raises(PermissionError):correlation.write(target,{'done':False})
    assert len(calls)==5 and correlation.read(target)=={'done':True}


def test_recupera_registro_temporario_sem_recalcular(tmp_path):
    steps=correlation.Steps(tmp_path,'contract',reserve_gib=0)
    output=tmp_path/'result.txt';output.write_text('value')
    marker=tmp_path/'etapas/day.json';partial=marker.with_name(marker.name+'.partial')
    correlation.write(marker,dict(contract='contract',status='running'))
    correlation.write(partial,dict(contract='contract',status='completed',outputs={str(output):correlation.sha(output)},result={'rows':3}))
    def forbidden():raise AssertionError('Não deve recalcular')
    assert steps.run('day',[output],forbidden)=={'rows':3}
    assert not partial.exists()


def test_compatibilidade_preserva_contrato_e_recusa_mudanca_cientifica():
    old=dict(code_sha256='1a13e2e1ba75279f0a7943163bf2229885e7cdf33361bc1f2ff0027e8fabe0bc',rows=10)
    prior=dict(identity=old,signature=correlation.signature(old))
    assert correlation.resume_identity(dict(old,code_sha256='patched'),prior)==old
    with pytest.raises(ValueError):correlation.resume_identity(dict(old,rows=11),prior)

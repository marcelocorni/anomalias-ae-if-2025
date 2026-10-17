import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sklearn.metrics import average_precision_score,auc,precision_recall_curve,roc_auc_score

from src.avaliacao import resumir,validar_alinhamento
from src.avaliacao_externa import avaliar,classificacao,integrar_grupos
from src.comum import parquet
from src.drift import psi
from src.labelcloud import criar_tags


def test_metricas_exatas_com_empates_conferem_com_sklearn():
    rng=np.random.default_rng(42)
    y=rng.integers(0,2,1000)
    scores=np.round(rng.normal(size=1000),1)
    groups=[(s,int(y[scores==s].sum()),int((scores==s).sum())) for s in sorted(set(scores),reverse=True)]
    result=integrar_grupos(groups,int(y.sum()),len(y),max_points=5)
    p,r,_=precision_recall_curve(y,scores)
    assert result['roc_auc']==pytest.approx(roc_auc_score(y,scores),abs=1e-12)
    assert result['average_precision']==pytest.approx(average_precision_score(y,scores),abs=1e-12)
    assert result['pr_auc_trapezoid']==pytest.approx(auc(r,p),abs=1e-12)
    assert len(result['curves'])<len(groups)


@pytest.mark.parametrize('positives',[0,10])
def test_uma_classe_retorna_metricas_indisponiveis(positives):
    result=integrar_grupos([(1.,positives,10)],positives,10)
    assert result['roc_auc'] is None and result['average_precision'] is None
    assert result['unavailable_reason']
    assert classificacao(0,0,0,10)['precision'] is None
    assert classificacao(0,0,0,10)['mcc'] is None


def test_labelcloud_join_sem_multiplicar_linhas_topk_e_jaccard(con,tmp_path):
    raw=tmp_path/'raw.parquet';scores=tmp_path/'scores.parquet';tags=tmp_path/'tags.parquet'
    pq.write_table(pa.table({'row_id':[1,2,3,4],'from_address':['0xAA','0xaa','0xbb','0xcc']}),raw)
    pq.write_table(pa.table({'row_id':[1,2,3,4],'score':np.array([.5,.5,.5,.1],dtype=np.float32)}),scores)
    pq.write_table(pa.table({'address':['0xaa','0xaa','0xbb'],'is_mev_bot':[True,False,False]}),tags)
    criar_tags(con,[tags])
    thresholds={'q990':.5,'q995':.5,'q999':1.}
    result=avaliar(con,[raw],[scores],thresholds,[.25,.5])
    assert result['rows']==4 and result['positives']==2
    assert result['top_k'][0]['associated']==1 and result['top_k'][0]['rows']==1
    assert result['top_k'][1]['associated']==2
    assert result['thresholds']['q995']['tp']==2 and result['thresholds']['q995']['fp']==1
    summary=resumir(con,[scores],[scores],thresholds,thresholds)
    assert all(r['jaccard']==1 for r in summary['agreement'].values())
    assert summary['agreement']['q999']['union']==0
    bad=tmp_path/'bad.parquet';pq.write_table(pa.table({'row_id':[1,1,3,4],'score':[1.,2.,3.,4.]}),bad)
    with pytest.raises(ValueError):validar_alinhamento(con,parquet([scores]),parquet([bad]))
    pq.write_table(pa.table({'row_id':[1,2,3],'score':[1.,2.,3.]}),bad)
    with pytest.raises(ValueError):validar_alinhamento(con,parquet([scores]),parquet([bad]))


def test_psi_identico_e_diferente():
    assert psi([.3,.7],[.3,.7])==0
    assert psi([.3,.7],[.7,.3])>0

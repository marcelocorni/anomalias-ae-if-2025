from __future__ import annotations

import math

from .avaliacao import validar_alinhamento
from .comum import dividir, mcc, parquet
from .labelcloud import ASSUMPTION


def integrar_grupos(groups, positives, total, max_points=4096):
    negatives = total-positives
    valid = 0 < positives < total
    tp = fp = count = 0
    auc = ap = pr_auc = 0.
    previous_tpr = previous_fpr = previous_recall = 0.
    previous_precision = 1.
    step = max(1, math.ceil(total/max_points))
    next_point = step
    curves = [dict(score=None,fpr=0.,tpr=0. if positives else None,recall=0. if positives else None,precision=1.,survival=0.)]
    last = None
    for score, p, n in groups:
        tp += int(p)
        fp += int(n)-int(p)
        count += int(n)
        tpr, fpr = dividir(tp,positives), dividir(fp,negatives)
        precision, recall = tp/count, tpr
        if valid:
            auc += (fpr-previous_fpr)*(tpr+previous_tpr)/2
            ap += (recall-previous_recall)*precision
            pr_auc += (recall-previous_recall)*(precision+previous_precision)/2
        previous_tpr, previous_fpr, previous_recall, previous_precision = tpr, fpr, recall, precision
        last = dict(score=float(score),fpr=fpr,tpr=tpr,recall=recall,precision=precision,survival=count/total)
        if count >= next_point:
            curves.append(last)
            next_point = count+step
    if count != total or tp != positives:
        raise ValueError('Contagem dos grupos de escores inconsistente.')
    if last is not None and curves[-1] != last:
        curves.append(last)
    return dict(rows=total,positives=positives,unassociated=negatives,prevalence=dividir(positives,total),
                roc_auc=auc if valid else None,average_precision=ap if valid else None,
                pr_auc_trapezoid=pr_auc if valid else None,
                unavailable_reason=None if valid else 'A janela não contém duas classes de associação.',
                curves=curves,curve_points_reduced_for_plot=True,metrics_use_all_score_groups=True)


def classificacao(tp,fp,fn,tn):
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=dividir(tp,tp+fp),recall=dividir(tp,tp+fn),
                f1=dividir(2*tp,2*tp+fp+fn),specificity=dividir(tn,tn+fp),mcc=mcc(tp,fp,fn,tn))


def avaliar(con, raw_files, score_files, thresholds, top_fractions):
    raw, scores = parquet(raw_files), parquet(score_files)
    validar_alinhamento(con,raw,scores)
    con.execute('DROP TABLE IF EXISTS auxiliary_evaluation')
    con.execute(f'''CREATE TEMP TABLE auxiliary_evaluation AS SELECT s.row_id,s.score,
        coalesce(t.is_mev_bot,false)::UTINYINT AS label
        FROM {scores} s JOIN {raw} m USING(row_id) LEFT JOIN tags t ON lower(m.from_address)=t.address''')
    total, positives, invalid = con.execute('SELECT count(*),sum(label),count(*) FILTER(WHERE score IS NULL OR NOT isfinite(score)) FROM auxiliary_evaluation').fetchone()
    if not total or invalid:
        raise ValueError('Avaliação externa vazia ou com escores inválidos.')
    groups_query = con.execute('SELECT score,sum(label)::BIGINT,count(*)::BIGINT FROM auxiliary_evaluation GROUP BY score ORDER BY score DESC')
    def groups():
        while batch := groups_query.fetchmany(8192):
            yield from batch
    ranking = integrar_grupos(groups(), positives, total)
    ranking['assumption'] = ASSUMPTION
    distributions = con.execute('''SELECT label,count(*),min(score),max(score),avg(score),
        quantile_cont(score,[0.25,0.5,0.75,0.99]) FROM auxiliary_evaluation GROUP BY label ORDER BY label''').fetchall()
    ranking['score_distributions_by_association'] = [dict(associated=bool(label),rows=n,min=low,max=high,mean=mean,
                                                          q25=qs[0],median=qs[1],q75=qs[2],q99=qs[3])
                                                    for label,n,low,high,mean,qs in distributions]
    ranking['thresholds'] = {}
    for name, threshold in thresholds.items():
        tp, fp, fn, tn = con.execute(f'''SELECT count(*) FILTER(WHERE label=1 AND score >= {threshold:.17g}),
            count(*) FILTER(WHERE label=0 AND score >= {threshold:.17g}),
            count(*) FILTER(WHERE label=1 AND score < {threshold:.17g}),
            count(*) FILTER(WHERE label=0 AND score < {threshold:.17g}) FROM auxiliary_evaluation''').fetchone()
        ranking['thresholds'][name] = classificacao(tp,fp,fn,tn)
    sizes = [max(1,math.ceil(total*f)) for f in top_fractions]
    expressions = ','.join(f'sum(label) FILTER(WHERE position <= {n})' for n in sizes)
    values = con.execute(f'''SELECT {expressions} FROM (SELECT label,
        row_number() OVER(ORDER BY score DESC,row_id ASC) AS position FROM auxiliary_evaluation)''').fetchone()
    ranking['top_k'] = [dict(fraction=f,rows=n,associated=int(p),precision=p/n,
                              recall=dividir(p,positives),lift=dividir(p/n,positives/total),tie_break='row_id ASC')
                           for f,n,p in zip(top_fractions,sizes,values)]
    con.execute('DROP TABLE auxiliary_evaluation')
    return ranking

from .comum import dividir, parquet


def validar_alinhamento(con, left, right):
    for relation in (left,right):
        n, ids = con.execute(f'SELECT count(*),count(DISTINCT row_id) FROM {relation}').fetchone()
        if not n or n != ids:
            raise ValueError('Chaves duplicadas ou conjunto vazio na avaliação.')
    missing = con.execute(f'SELECT count(*) FROM {left} a FULL OUTER JOIN {right} b USING(row_id) WHERE a.row_id IS NULL OR b.row_id IS NULL').fetchone()[0]
    if missing:
        raise ValueError(f'Cobertura incompleta no alinhamento: {missing} chaves.')


def resumir(con, ae_files, if_files, ae_thresholds, if_thresholds):
    ae, iff = parquet(ae_files), parquet(if_files)
    validar_alinhamento(con,ae,iff)
    output = dict(models={}, agreement={})
    for method, relation, thresholds in [('AE',ae,ae_thresholds),('IF',iff,if_thresholds)]:
        n, low, high, mean, qs = con.execute(f'SELECT count(*),min(score),max(score),avg(score),quantile_cont(score,[0.25,0.5,0.75,0.95,0.99,0.995,0.999]) FROM {relation}').fetchone()
        rates = {}
        for key, threshold in thresholds.items():
            count = con.execute(f'SELECT count(*) FROM {relation} WHERE score >= {threshold:.17g}').fetchone()[0]
            rates[key] = dict(threshold=threshold, flagged=count, rate=count/n)
        output['models'][method] = dict(rows=n,min=low,max=high,mean=mean,quantiles=dict(zip(['q250','q500','q750','q950','q990','q995','q999'],qs)),rates=rates)
    for key in ae_thresholds:
        a, b = ae_thresholds[key], if_thresholds[key]
        intersection, union = con.execute(f'''SELECT count(*) FILTER(WHERE a.score >= {a:.17g} AND b.score >= {b:.17g}),
            count(*) FILTER(WHERE a.score >= {a:.17g} OR b.score >= {b:.17g})
            FROM {ae} a JOIN {iff} b USING(row_id)''').fetchone()
        output['agreement'][key] = dict(intersection=intersection,union=union,jaccard=dividir(intersection,union) if union else 1., empty_sets_convention=1.)
    return output

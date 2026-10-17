import math

from .comum import assinatura, parquet


def calibrar(con, files, role, quantiles, model_hash, input_signature):
    if role != 'validation':
        raise ValueError('Limiares só podem ser calibrados na validação temporal.')
    relation = parquet(files)
    count, invalid = con.execute(f'SELECT count(*),count(*) FILTER(WHERE score IS NULL OR NOT isfinite(score)) FROM {relation}').fetchone()
    if not count or invalid:
        raise ValueError('Calibração vazia ou com escores inválidos.')
    values = con.execute(f'SELECT quantile_cont(score, {list(quantiles.values())}) FROM {relation}').fetchone()[0]
    thresholds = dict(zip(quantiles, map(float, values)))
    if not all(math.isfinite(v) for v in thresholds.values()):
        raise ValueError('Limiares inválidos.')
    document = dict(role='validation', quantiles=quantiles, thresholds=thresholds, rows=count, model_sha256=model_hash, input_signature=input_signature, comparison='>=')
    document['signature'] = assinatura(document)
    return document

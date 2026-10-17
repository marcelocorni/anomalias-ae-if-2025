from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from .atributos import FEATURES
from .comum import ident, parquet


def treinar(con, files, settings, seed, threads, output, contract):
    relation = parquet(files)
    total = con.execute(f'SELECT count(*) FROM {relation}').fetchone()[0]
    if not total:
        raise ValueError('Treino IF vazio.')
    target = min(total, settings['training_sample_size'])
    where = f'WHERE hash(row_id) % {total} < {target}' if target < total else ''
    columns = ','.join(ident(f) for f in FEATURES)
    arrays = con.execute(f'SELECT {columns} FROM {relation} {where} ORDER BY row_id').fetchnumpy()
    values = np.column_stack([arrays[f] for f in FEATURES]).astype(np.float32, copy=False)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Amostra IF vazia ou inválida.')
    model = IsolationForest(n_estimators=settings['n_estimators'], max_samples=min(settings['max_samples_tree'], len(values)),
                            max_features=1., contamination='auto', bootstrap=False, random_state=seed, n_jobs=threads)
    model.fit(values)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.partial')
    joblib.dump(dict(estimator=model, features=FEATURES, contract=contract), temp, compress=3)
    temp.replace(path)
    return dict(training_rows_total=total, training_rows_sample=len(values), requested=target, fraction=len(values)/total,
                max_samples_per_tree=model.max_samples_, rule='hash(row_id) % training_rows_total < min(training_rows_total, requested)')


def carregar(path, threads):
    state = joblib.load(path)
    if state['features'] != FEATURES:
        raise ValueError('Contrato B incompatível no modelo IF.')
    def score(values):
        with joblib.parallel_backend('threading', n_jobs=threads):
            return -state['estimator'].score_samples(values)
    return score

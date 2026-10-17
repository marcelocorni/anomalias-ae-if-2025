from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .atributos import FEATURES


def lotes(files, batch_size):
    for path in files:
        source = pq.ParquetFile(path)
        for batch in source.iter_batches(batch_size=batch_size, columns=['row_id', *FEATURES]):
            ids = batch.column(0).to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)
            values = np.column_stack([batch.column(i+1).to_numpy(zero_copy_only=False) for i in range(len(FEATURES))]).astype(np.float32, copy=False)
            if not np.isfinite(values).all():
                raise ValueError('Entrada do modelo não finita ou fora do alcance float32.')
            yield ids, values


def gravar_escores(files, output, scorer, batch_size):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + '.partial')
    writer = None
    rows = 0
    try:
        for ids, values in lotes(files, batch_size):
            scores = np.asarray(scorer(values), dtype=np.float32)
            if scores.shape != (len(ids),) or not np.isfinite(scores).all():
                raise ValueError('Escores inválidos ou cobertura incompatível.')
            table = pa.table({'row_id': ids, 'score': scores})
            if writer is None:
                writer = pq.ParquetWriter(temp, table.schema, compression='zstd')
            writer.write_table(table)
            rows += len(ids)
    finally:
        if writer is not None:
            writer.close()
    if not rows:
        raise ValueError('Inferência sem registros.')
    temp.replace(output)
    return dict(rows=rows)

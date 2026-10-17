"""Conferência independente de leitura; comparações não reutilizam resultados."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .comum import sha256, conectar, parquet


def conferir(path):
    path = Path(path)
    before = sha256(path)
    result = dict(path=str(path), sha256=before, bytes=path.stat().st_size)
    if path.suffix == '.parquet':
        source = pq.ParquetFile(path)
        rows = 0
        for batch in source.iter_batches(batch_size=65536, use_threads=False):
            batch.validate(full=True)
            rows += batch.num_rows
        if rows != source.metadata.num_rows:
            raise ValueError(f'Contagem de leitura diverge do rodapé: {path}')
        result.update(rows=rows, schema=str(source.schema_arrow))
    after = sha256(path)
    if before != after:
        raise ValueError(f'Leituras SHA-256 inconsistentes: {path}')
    return result


def comparar(path, run, old_runs):
    path, run = Path(path), Path(run)
    relative = path.relative_to(run)
    current = sha256(path)
    results = []
    for old in old_runs:
        previous = run.parent / old / relative
        if not previous.is_file():
            results.append(dict(run=old, artifact=str(relative), status='absent'))
            continue
        before = sha256(previous)
        if before != sha256(previous):
            raise ValueError(f'Leitura inconsistente do artefato preservado: {previous}')
        item = dict(run=old, artifact=str(relative), previous_sha256=before,
                            regenerated_sha256=current,
                            status='identical_bytes' if before == current else 'different_bytes',
                            note='Nenhum resultado antigo foi reutilizado. Diferenças não identificam sua causa.')
        if before != current and path.suffix == '.parquet':
            try:
                conferir(previous)
            except (pa.ArrowInvalid, OSError) as exc:
                if sha256(previous) != before:
                    raise ValueError(f'Leitura inconsistente do artefato preservado: {previous}') from exc
                item.update(status='previous_unreadable', error=str(exc))
                results.append(item)
                continue
            a, b = pq.ParquetFile(path), pq.ParquetFile(previous)
            item.update(regenerated_rows=a.metadata.num_rows,previous_rows=b.metadata.num_rows)
            if a.schema_arrow.equals(b.schema_arrow,check_metadata=False):
                with conectar(run/'.tmp/comparacao','2GB',2) as con:
                    current_table, old_table = parquet([path]), parquet([previous])
                    item['regenerated_minus_previous'] = con.execute(f'SELECT count(*) FROM (SELECT * FROM {current_table} EXCEPT ALL SELECT * FROM {old_table})').fetchone()[0]
                    item['previous_minus_regenerated'] = con.execute(f'SELECT count(*) FROM (SELECT * FROM {old_table} EXCEPT ALL SELECT * FROM {current_table})').fetchone()[0]
            else:
                item['schema_equal'] = False
        results.append(item)
    return results

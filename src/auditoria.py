from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pyarrow.parquet as pq

from .comum import ident, parquet, q


def arquivos_do_dia(root, day):
    files = sorted((Path(root) / f'date={day}').glob('*.parquet'))
    if not files:
        raise FileNotFoundError(f'Dia sem Parquets: {day}')
    return files


def auditar_raw(con, day):
    start = datetime.fromisoformat(day)
    end = start + timedelta(days=1)
    invalid = con.execute(f"""SELECT count(*) FROM raw_transactions
        WHERE block_number IS NULL OR block_number < 0 OR block_number >= 4294967296
           OR transaction_index IS NULL OR transaction_index < 0 OR transaction_index >= 4294967296
           OR hash IS NULL OR NOT regexp_full_match(hash, '0x[0-9a-fA-F]{{64}}')
           OR block_hash IS NULL OR NOT regexp_full_match(block_hash, '0x[0-9a-fA-F]{{64}}')
           OR date IS NULL OR CAST(date AS VARCHAR) <> {q(day)}
           OR block_timestamp IS NULL OR block_timestamp < TIMESTAMP {q(start)} OR block_timestamp >= TIMESTAMP {q(end)}
           OR transaction_type IS NULL OR receipt_status IS NULL OR receipt_status NOT IN (0,1)
        """).fetchone()[0]
    if invalid:
        raise ValueError(f'{day}: {invalid} registros com identificadores, tipos ou datas inválidos.')
    rows, blocks, first, last, unique_hashes = con.execute('SELECT count(*),count(DISTINCT block_number),min(block_number),max(block_number),count(DISTINCT lower(hash)) FROM raw_transactions').fetchone()
    if not rows:
        raise ValueError(f'{day}: arquivo diário vazio; conferir cobertura externamente.')
    if rows != unique_hashes:
        raise ValueError(f'{day}: hashes duplicados.')
    bad_block = con.execute('''SELECT block_number FROM raw_transactions GROUP BY block_number
        HAVING count(*) <> count(DISTINCT transaction_index)
           OR min(transaction_index) <> 0 OR max(transaction_index) <> count(*)-1
           OR count(DISTINCT block_hash) <> 1 OR count(DISTINCT block_timestamp) <> 1 LIMIT 1''').fetchone()
    if bad_block:
        raise ValueError(f'{day}: bloco inconsistente ou posições incompletas: {bad_block[0]}')
    return dict(rows=rows, blocks=blocks, first_block=first, last_block=last,
                blockchain_coverage_verified=False,
                limitation='Sem reconciliação externa: blocos vazios/ausentes e sufixos de transações ausentes não são distinguíveis.')


def validar_numericos(con, relation, features):
    conditions = ' OR '.join(f'{ident(f)} IS NULL OR NOT isfinite(CAST({ident(f)} AS DOUBLE))' for f in features)
    invalid = con.execute(f'SELECT count(*) FROM {relation} WHERE {conditions}').fetchone()[0]
    if invalid:
        raise ValueError(f'{invalid} registros com atributos nulos ou não finitos; nenhuma imputação automática.')


def validar_chaves(con, relation):
    n, ids = con.execute(f'SELECT count(*), count(DISTINCT row_id) FROM {relation}').fetchone()
    if not n or n != ids:
        raise ValueError('Chaves duplicadas ou matriz vazia.')
    return n


def auditar_global(con, files):
    relation = parquet(files)
    duplicate = con.execute(f'SELECT lower(hash) FROM {relation} GROUP BY lower(hash) HAVING count(*)>1 LIMIT 1').fetchone()
    if duplicate:
        raise ValueError(f'Hash repetido entre partições: {duplicate[0]}')
    duplicate_id = con.execute(f'SELECT row_id FROM {relation} GROUP BY row_id HAVING count(*)>1 LIMIT 1').fetchone()
    if duplicate_id:
        raise ValueError('Bloco/posição repetido entre partições.')
    blocks, low, high = con.execute(f'SELECT count(DISTINCT block_number),min(block_number),max(block_number) FROM {relation}').fetchone()
    gaps = con.execute(f'''SELECT previous+1 AS first_missing,block_number-1 AS last_missing
        FROM (SELECT block_number,lag(block_number) OVER(ORDER BY block_number) AS previous
        FROM (SELECT DISTINCT block_number FROM {relation}))
        WHERE block_number>previous+1 ORDER BY block_number''').fetchall()
    return dict(rows=sum(pq.ParquetFile(f).metadata.num_rows for f in files), duplicate_hashes=0,
                duplicate_ids=0, observed_blocks=blocks, first_block=low, last_block=high,
                unobserved_block_ranges=[list(pair) for pair in gaps],
                unobserved_block_numbers=sum(b-a+1 for a,b in gaps),
                gap_note='Intervalo da seleção; em ensaios pode incluir dias não selecionados. Não distingue blocos vazios de ausentes.',
                blockchain_coverage_verified=False)

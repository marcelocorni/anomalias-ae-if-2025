import csv
import re
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

from .comum import parquet, sha256

CATEGORY_RANGES = [(1,163,'mev_builder'), (164,3911,'mev_bot'),
                   (3912,3912,'mev_relay'), (3913,3913,'mev_protection')]
ASSUMPTION = 'Associação histórica do remetente a mev_bot=1; ausência dessa associação=0, sem afirmar benignidade ou ataque.'


def normalizar(source, settings, output):
    if sha256(source) != settings['sha256']:
        raise ValueError('CSV Label Cloud diferente do snapshot auditado.')
    with source.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['transaction_from', 'label']:
            raise ValueError('Cabeçalho Label Cloud incompatível.')
        raw = list(reader)
    if len(raw) != settings['expected_rows'] or len(raw) != 3913:
        raise ValueError('Quantidade de rótulos diferente da reconstrução histórica.')
    rows = []
    for position, record in enumerate(raw, 1):
        address = record['transaction_from'].strip().lower()
        if not re.fullmatch(r'0x[0-9a-f]{40}', address):
            raise ValueError(f'Endereço Label Cloud inválido na linha {position}.')
        category = next(c for a,b,c in CATEGORY_RANGES if a<=position<=b)
        rows.append(dict(address=address, name_tag=record['label'].strip(), category=category,
                         is_mev_bot=category=='mev_bot', source_position=position))
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name+'.partial')
    pq.write_table(pa.Table.from_pylist(rows), temp, compression='zstd')
    temp.replace(output)
    return dict(source_sha256=settings['sha256'], snapshot_date=settings['snapshot_date'], rows=len(rows),
                unique_addresses=len({r['address'] for r in rows}),
                exact_duplicate_rows=len(raw)-len({(r['transaction_from'].lower(),r['label']) for r in raw}),
                categories=dict(Counter(r['category'] for r in rows)),
                category_basis='positional_reconstruction', join_key='lower(from_address)=address', assumption=ASSUMPTION)


def criar_tags(con, files):
    con.execute(f'''CREATE OR REPLACE VIEW tags AS SELECT lower(address) AS address,
        bool_or(is_mev_bot) AS is_mev_bot FROM {parquet(files)} GROUP BY lower(address)''')

import copy
import csv
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.comum import conectar, ler_json
from src.configuracao import RAIZ


@pytest.fixture
def config():
    c = copy.deepcopy(ler_json(RAIZ/'configuracao/experimento.json'))
    c['autoencoder'].update(device='cpu',amp=False,max_epochs=2,batch_size=16,patience=2)
    c['isolation_forest'].update(n_estimators=5,max_samples_tree=16,training_sample_size=30)
    c['recursos'].update(threads=1,memory_limit='512MB',min_free_gib=0,inference_batch_size=11)
    c['drift'].update(ks_grid_points=5,psi_bins=3)
    return c


@pytest.fixture
def con(tmp_path):
    with conectar(tmp_path/'duckdb','512MB',1) as connection:
        yield connection


@pytest.fixture
def raw_factory():
    with (RAIZ/'dados_auxiliares/labelcloud.csv').open(encoding='utf-8-sig',newline='') as stream:
        records = list(csv.DictReader(stream))
    known_bot = records[163]['transaction_from'].strip().lower()
    def make(day='2025-01-01',start_block=21525891,blocks=12):
        records = []
        for b in range(blocks):
            number = start_block+b
            for index in range(1+b%5):
                kind = (b+index)%5
                price = 1_000_000_000+(b+index)*1000
                records.append(dict(hash='0x'+f'{number*1000+index:064x}',block_hash='0x'+f'{number:064x}',
                    block_number=number,transaction_index=index,block_timestamp=datetime.fromisoformat(day)+timedelta(seconds=b*12),date=day,
                    from_address=known_bot if index%2==0 else '0x'+'a'*40,
                    to_address=None if index%3==0 else '0x'+'b'*40,
                    gas=30000+b*2000,receipt_gas_used=21000+index*500,
                    value=float((index+b)*10**15),nonce=b*3+index,gas_price=price,
                    receipt_effective_gas_price=None if b%4==0 else price,
                    max_fee_per_gas=0 if kind in (0,1) else price*2,
                    max_priority_fee_per_gas=0 if kind in (0,1) else 10000,
                    receipt_status=(b+index)%2,transaction_type=kind))
        return records
    return make


@pytest.fixture
def write_raw(tmp_path,raw_factory):
    def write(day='2025-01-01',rows=None,name='raw.parquet'):
        path=tmp_path/name
        path.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(pa.Table.from_pylist(raw_factory(day) if rows is None else rows),path)
        return path
    return write

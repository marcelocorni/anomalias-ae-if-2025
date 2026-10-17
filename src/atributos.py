from __future__ import annotations

import duckdb

from .comum import exportar, parquet, q

FEATURES = ['gas_used_log', 'gas_used_ratio', 'effective_gas_price_log', 'max_fee_per_gas_log', 'max_priority_fee_log', 'fee_cap_utilization', 'transaction_value_log', 'nonce_log', 'receipt_status', 'tx_type_0', 'tx_type_1', 'tx_type_3', 'transaction_index_pct', 'block_tx_count_log', 'sender_tx_count_block_log', 'receiver_tx_count_block_log', 'gas_price_rank_block', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos']
METADATA = ["row_id", "hash", "block_hash", "block_number", "block_timestamp", "date", "from_address", "to_address", "transaction_index", "transaction_type"]

def validate_schema(con: duckdb.DuckDBPyConnection) -> None:
    schema = con.execute("DESCRIBE SELECT * FROM raw_transactions").fetchall()
    available = {row[0] for row in schema}
    required = {
        "hash",
        "block_hash",
        "block_number",
        "block_timestamp",
        "date",
        "from_address",
        "to_address",
        "transaction_index",
        "gas",
        "value",
        "gas_price",
        "receipt_gas_used",
        "receipt_status",
        "transaction_type",
        "max_priority_fee_per_gas",
        "nonce",
        "receipt_effective_gas_price",
        "max_fee_per_gas",
    }
    missing = sorted(required - available)
    if missing:
        raise RuntimeError("Colunas obrigatórias ausentes: " + ", ".join(missing))

def create_feature_master(
    con: duckdb.DuckDBPyConnection
) -> None:
    con.execute(
        f"""
        CREATE TEMP TABLE feature_master AS
        WITH sampled_blocks AS (
            SELECT
                hash,
                block_hash,
                block_number,
                block_timestamp,
                date,
                from_address,
                to_address,
                transaction_index,
                transaction_type,
                gas,
                value,
                receipt_gas_used,
                receipt_status,
                max_priority_fee_per_gas,
                nonce,
                COALESCE(receipt_effective_gas_price, gas_price, 0)
                    AS effective_price,
                max_fee_per_gas,
                COUNT(*) OVER (PARTITION BY block_number) AS block_tx_count,
                COUNT(*) OVER (
                    PARTITION BY block_number, from_address
                ) AS sender_tx_count_block,
                COUNT(*) OVER (
                    PARTITION BY block_number, to_address
                ) AS receiver_tx_count_block,
                PERCENT_RANK() OVER (
                    PARTITION BY block_number
                    ORDER BY COALESCE(receipt_effective_gas_price, gas_price, 0)
                ) AS gas_price_rank_block
            FROM raw_transactions
            WHERE block_number % 1 = 0
        ),
        engineered AS (
            SELECT
                hash,
                block_hash,
                block_number,
                block_timestamp,
                date,
                from_address,
                to_address,
                transaction_index,
                transaction_type,
                LN(1.0 + GREATEST(COALESCE(receipt_gas_used, 0), 0))
                    AS gas_used_log,
                CASE
                    WHEN gas > 0
                    THEN CAST(receipt_gas_used AS DOUBLE) / CAST(gas AS DOUBLE)
                    ELSE 0.0
                END AS gas_used_ratio,
                LN(1.0 + GREATEST(CAST(effective_price AS DOUBLE), 0.0))
                    AS effective_gas_price_log,
                LN(
                    1.0 + GREATEST(CAST(COALESCE(max_fee_per_gas, 0) AS DOUBLE), 0.0)
                ) AS max_fee_per_gas_log,
                LN(
                    1.0
                    + GREATEST(
                        CAST(COALESCE(max_priority_fee_per_gas, 0) AS DOUBLE),
                        0.0
                    )
                ) AS max_priority_fee_log,
                CASE
                    WHEN max_fee_per_gas > 0
                    THEN CAST(effective_price AS DOUBLE)
                         / CAST(max_fee_per_gas AS DOUBLE)
                    ELSE 0.0
                END AS fee_cap_utilization,
                LN(1.0 + GREATEST(CAST(COALESCE(value, 0) AS DOUBLE), 0.0))
                    AS transaction_value_log,
                LN(1.0 + GREATEST(CAST(COALESCE(nonce, 0) AS DOUBLE), 0.0))
                    AS nonce_log,
                CAST(COALESCE(receipt_status, 0) AS UTINYINT) AS receipt_status,
                CAST(transaction_type = 0 AS UTINYINT) AS tx_type_0,
                CAST(transaction_type = 1 AS UTINYINT) AS tx_type_1,
                CAST(transaction_type = 3 AS UTINYINT) AS tx_type_3,
                CASE
                    WHEN block_tx_count > 1
                    THEN CAST(transaction_index AS DOUBLE) / (block_tx_count - 1)
                    ELSE 0.0
                END AS transaction_index_pct,
                LN(1.0 + block_tx_count) AS block_tx_count_log,
                LN(1.0 + sender_tx_count_block) AS sender_tx_count_block_log,
                LN(1.0 + receiver_tx_count_block) AS receiver_tx_count_block_log,
                gas_price_rank_block,
                SIN(2.0 * PI() * EXTRACT(hour FROM block_timestamp) / 24.0)
                    AS hour_sin,
                COS(2.0 * PI() * EXTRACT(hour FROM block_timestamp) / 24.0)
                    AS hour_cos,
                SIN(2.0 * PI() * EXTRACT(dow FROM block_timestamp) / 7.0)
                    AS weekday_sin,
                COS(2.0 * PI() * EXTRACT(dow FROM block_timestamp) / 7.0)
                    AS weekday_cos
            FROM sampled_blocks
        )
        SELECT
            ((block_number::UBIGINT << 32) | transaction_index::UBIGINT) AS row_id,
            *
        FROM engineered
        """
    )


def gerar(con, files, day, output):
    from .auditoria import auditar_raw, validar_chaves, validar_numericos
    con.execute('DROP TABLE IF EXISTS feature_master')
    con.execute('CREATE OR REPLACE VIEW raw_transactions AS SELECT * FROM ' + parquet(files))
    validate_schema(con)
    audit = auditar_raw(con, day)
    create_feature_master(con)
    validar_chaves(con, 'feature_master')
    validar_numericos(con, 'feature_master', FEATURES)
    columns = ','.join(METADATA + [f'CAST({f} AS DOUBLE) AS {f}' for f in FEATURES])
    exportar(con, f'SELECT {columns} FROM feature_master ORDER BY row_id', output)
    con.execute('DROP TABLE feature_master')
    return audit

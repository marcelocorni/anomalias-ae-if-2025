"""Função congelada para comparação da semântica B; sem entrada de execução."""
import duckdb

def create_feature_master(
    con: duckdb.DuckDBPyConnection, block_modulus: int
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
            WHERE block_number % {block_modulus} = 0
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
                    AS weekday_cos,
                LN(1.0 + GREATEST(CAST(COALESCE(gas, 0) AS DOUBLE), 0.0))
                    AS gas_limit_log,
                LN(
                    1.0
                    + GREATEST(CAST(COALESCE(receipt_gas_used, 0) AS DOUBLE), 0.0)
                      * GREATEST(CAST(effective_price AS DOUBLE), 0.0)
                ) AS transaction_fee_paid_log,
                CAST(
                    to_address IS NULL
                    OR TRIM(COALESCE(to_address, '')) IN ('', '0x')
                    AS UTINYINT
                ) AS is_contract_creation
            FROM sampled_blocks
        )
        SELECT
            ROW_NUMBER() OVER ()::UBIGINT AS row_id,
            *
        FROM engineered
        """
    )

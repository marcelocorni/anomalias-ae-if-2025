from __future__ import annotations

import math

from .atributos import FEATURES
from .auditoria import validar_chaves, validar_numericos
from .comum import assinatura, exportar, ident, parquet

ROBUST = ['gas_used_log', 'effective_gas_price_log', 'max_fee_per_gas_log',
          'max_priority_fee_log', 'transaction_value_log', 'nonce_log',
          'block_tx_count_log', 'sender_tx_count_block_log', 'receiver_tx_count_block_log']
CONDITIONAL = {'max_fee_per_gas_log': 'tx_type_0=0 AND tx_type_1=0',
               'max_priority_fee_log': 'tx_type_0=0 AND tx_type_1=0'}


def ajustar(con, files, role, input_signature):
    if role != 'train':
        raise ValueError('O pré-processamento só pode ser ajustado no treino.')
    relation = parquet(files)
    params = {}
    # Um atributo por consulta evita acumular os vetores de quantis de todos os atributos.
    for feature in ROBUST:
        condition = CONDITIONAL.get(feature)
        where = f'WHERE {condition}' if condition else ''
        quantiles = con.execute(f'SELECT quantile_cont({ident(feature)}, [0.25,0.5,0.75]) FROM {relation} {where}').fetchone()[0]
        if not quantiles or any(x is None or not math.isfinite(x) for x in quantiles):
            raise ValueError(f'Treino sem quantis válidos: {feature}')
        q1, median, q3 = map(float, quantiles)
        params[feature] = dict(q1=q1, median=median, q3=q3, scale=q3-q1 if q3>q1 else 1., fit_filter=condition)
    document = dict(features=FEATURES, fit_role='train', input_signature=input_signature, parameters=params)
    document['signature'] = assinatura(document)
    return document


def expressao(feature, params):
    if feature not in params:
        return f'CAST({ident(feature)} AS DOUBLE) AS {ident(feature)}'
    p = params[feature]
    value = f'((CAST({ident(feature)} AS DOUBLE)-({p["median"]:.17g}))/({p["scale"]:.17g}))'
    if p['fit_filter']:
        value = f'CASE WHEN {p["fit_filter"]} THEN {value} ELSE 0.0 END'
    return value + f' AS {ident(feature)}'


def transformar(con, files, document, output):
    if document['features'] != FEATURES or document['fit_role'] != 'train':
        raise ValueError('Contrato de pré-processamento incompatível.')
    unsigned = {k:v for k,v in document.items() if k != 'signature'}
    if assinatura(unsigned) != document['signature']:
        raise ValueError('Assinatura de pré-processamento inválida.')
    columns = ','.join(['row_id'] + [expressao(f, document['parameters']) for f in FEATURES])
    con.execute(f'CREATE OR REPLACE VIEW transformed AS SELECT {columns} FROM {parquet(files)}')
    count = validar_chaves(con, 'transformed')
    validar_numericos(con, 'transformed', FEATURES)
    exportar(con, 'SELECT * FROM transformed ORDER BY row_id', output)
    return dict(rows=count, preprocessing_signature=document['signature'])

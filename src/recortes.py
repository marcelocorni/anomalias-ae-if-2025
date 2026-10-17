from datetime import date, timedelta
import re


def validar_janelas(janelas):
    if any(not re.fullmatch('[a-z][a-z0-9_]*',j['chave']) for j in janelas):
        raise ValueError('Chave temporal inválida.')
    if len({j['chave'] for j in janelas}) != len(janelas):
        raise ValueError('Chaves temporais repetidas.')
    for role in ('train', 'validation', 'test'):
        if sum(j['papel'] == role for j in janelas) != 1:
            raise ValueError(f'É necessária uma janela {role}.')
    ordered = sorted(janelas, key=lambda j: j['inicio'])
    for n, window in enumerate(ordered):
        start, end = date.fromisoformat(window['inicio']), date.fromisoformat(window['fim_exclusivo'])
        if start >= end:
            raise ValueError('Janela temporal vazia ou invertida.')
        if n and ordered[n - 1]['fim_exclusivo'] != window['inicio']:
            raise ValueError('Janelas com lacuna ou sobreposição.')
    if ordered[0]['inicio'] != '2025-01-01' or ordered[-1]['fim_exclusivo'] != '2026-01-01':
        raise ValueError('As janelas devem cobrir 2025.')
    train, val, test = (next(j for j in janelas if j['papel'] == role) for role in ('train', 'validation', 'test'))
    if train['fim_exclusivo'] > val['inicio'] or val['fim_exclusivo'] > test['inicio']:
        raise ValueError('Treino, calibração e teste fora da ordem cronológica.')


def janela_do_dia(day, janelas):
    day = date.fromisoformat(str(day)).isoformat()
    matches = [j for j in janelas if j['inicio'] <= day < j['fim_exclusivo']]
    if len(matches) != 1:
        raise ValueError(f'Dia sem janela única: {day}')
    return matches[0]


def dias_do_ano():
    return [(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(365)]

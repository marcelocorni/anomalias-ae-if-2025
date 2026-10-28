"""Planeja ou importa etapas independentes dos modelos, sem iniciar o pipeline."""
import argparse
import json
import sys
from pathlib import Path

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from src.comum import ler_json
from src.configuracao import RAIZ,carregar
from src.pipeline import selecionar
from src.reutilizacao import planejar,executar


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origem',type=Path,default=Path('I:/arquivo-sbrc-2027/anual-2025-b-03'))
    parser.add_argument('--perfil',type=Path,default=Path('configuracao/execucao_revisao_01.json'))
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--verificar',action='store_true')
    group.add_argument('--executar',action='store_true')
    args=parser.parse_args()
    try:
        settings=ler_json(RAIZ/args.perfil)
        config,paths=carregar(RAIZ/settings['config'],RAIZ/settings['local'])
        records=selecionar(config,paths,'completo')
        print('Conferindo contratos e metadados da preparação...',flush=True)
        plan=planejar(RAIZ,args.origem,config,paths,settings['id'],records)
        if plan['rows']!=settings['expected_rows'] or sum(len(r['files']) for r in records)!=settings['expected_files']:
            raise ValueError('Inventário importado diferente do perfil aprovado.')
        print(json.dumps(dict(source=str(plan['source']),run=str(plan['target']),days=plan['days'],rows=plan['rows'],
            stages=len(plan['stages']),files=plan['files'],copy_gib=plan['bytes']/2**30,
            remaining_copy_gib=plan['remaining_bytes']/2**30,temporary_copy_gib=plan['temporary_bytes']/2**30,
            free_gib=plan['free_gib'],
            free_after_copy_gib=plan['free_after_copy_gib'],minimum_free_gib=config['recursos']['min_free_gib'],
            source_preserved=True,models_reused=False,training_started=False,pipeline_started=False,
            copy_started=False),ensure_ascii=False,indent=2),flush=True)
        if args.executar:
            executar(plan,config,paths,settings['id'])
        return 0
    except (ValueError,OSError,RuntimeError,KeyError) as exc:
        print(f'Importação interrompida: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        return 1


if __name__=='__main__':
    raise SystemExit(main())

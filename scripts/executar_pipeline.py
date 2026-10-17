import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from src.configuracao import carregar
from src.pipeline import PHASES, executar, planejar


def main():
    parser=argparse.ArgumentParser(description='Pipeline B: planejamento por padrão; execução explícita por fase.')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--local',type=Path)
    parser.add_argument('--modo',choices=['ensaio','completo'],default='completo')
    parser.add_argument('--dias',nargs='+')
    parser.add_argument('--execucao',help='Identificador novo ou execução compatível a retomar.')
    parser.add_argument('--ate',choices=PHASES,default='relatorios')
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--planejar',action='store_true')
    action.add_argument('--executar',action='store_true')
    args=parser.parse_args()
    try:
        config,paths=carregar(args.config,args.local)
        if not args.executar:
            print(json.dumps(planejar(config,paths,args.modo,args.dias),ensure_ascii=False,indent=2))
        else:
            result=executar(config,paths,args.modo,args.dias,args.execucao,args.ate)
            print(f'Execução gravada em {result}')
    except (ValueError,FileNotFoundError,RuntimeError) as exc:
        parser.exit(1,str(exc)+'\n')


if __name__=='__main__':
    main()

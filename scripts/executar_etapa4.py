import argparse
import json
import sys
import subprocess
from pathlib import Path

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from src.comum import ler_json,sha256
from src.configuracao import RAIZ,carregar
from src.operacao import conferir_arquivos,monitorar,processo_ativo
from src.pipeline import selecionar


def preparar():
    import psutil
    settings=ler_json(RAIZ/'configuracao/execucao_etapa4.json')
    config_path=RAIZ/settings['config'];local_path=RAIZ/settings['local']
    config,paths=carregar(config_path,local_path)
    selected=selecionar(config,paths,'completo')
    inventory=conferir_arquivos([p for day in selected for p in day['files']],settings['expected_files'],settings['expected_rows'])
    run=paths['resultados']/settings['id']
    if (run/'.lock').exists() and processo_ativo(run/'.lock'):
        raise RuntimeError('O pipeline anual já está ativo.')
    manifest=ler_json(run/'manifesto.json') if (run/'manifesto.json').exists() else None
    minimum=config['recursos']['min_free_gib'] if manifest else max(settings['initial_free_gib'],config['recursos']['min_free_gib'])
    free=psutil.disk_usage(str(RAIZ)).free
    if free < minimum*2**30:
        raise ValueError(f'Espaço livre insuficiente: {free/2**30:.1f} GiB; necessário {minimum} GiB.')
    probe = subprocess.run([sys.executable,'-X','utf8','-c',
        "import torch,json; ok=torch.cuda.is_available(); print(json.dumps({'available':ok,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if ok else None}))"],
        capture_output=True,text=True,encoding='utf-8',timeout=45,check=True)
    gpu = json.loads(probe.stdout)
    if config['autoencoder']['device']!='cuda' or not gpu['available']:
        raise RuntimeError('CUDA não está disponível; a execução anual não será iniciada em CPU.')
    command=[str(RAIZ/'.venv/Scripts/python.exe'),'-u',str(RAIZ/'scripts/executar_pipeline.py'),
             '--config',str(config_path),'--local',str(local_path),'--modo','completo','--execucao',settings['id'],'--ate','relatorios','--executar']
    return settings,paths,run,command,dict(id=settings['id'],mode='completo',inventory=inventory,days=len(selected),
        config=str(config_path),config_sha256=sha256(config_path),free_gib=free/2**30,minimum_free_gib=minimum,
        cuda=gpu['cuda'],gpu=gpu['gpu'],
        existing_status=manifest['status'] if manifest else None,annual_execution_started=False,
        run=str(run),operation_logs=str(run/'operacao'))


def main():
    parser=argparse.ArgumentParser(description='Etapa 4: verificar ou executar/retomar o ano completo.')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--verificar',action='store_true')
    group.add_argument('--executar',action='store_true')
    args=parser.parse_args()
    try:
        settings,paths,run,command,report=preparar()
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
        if args.executar:
            return monitorar(command,RAIZ,run,run/'operacao',
                             settings['monitor_interval_seconds'],settings['prevent_sleep'])
        return 0
    except (ValueError,OSError,RuntimeError,subprocess.SubprocessError) as exc:
        print(str(exc),file=sys.stderr,flush=True)
        return 1


if __name__=='__main__':
    raise SystemExit(main())

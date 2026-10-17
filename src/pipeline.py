from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from . import atributos, auditoria, avaliacao, avaliacao_externa, calibracao, drift, labelcloud, preprocessamento
from .comum import conectar, escrever_json, ler_json
from .configuracao import RAIZ
from .dados import gravar_escores
from .execucao import Execucao
from .recortes import dias_do_ano, janela_do_dia

PHASES = ['atributos','preprocessamento','modelos','calibracao','avaliacao','relatorios']


def selecionar(config, paths, mode, days=None):
    if mode not in ('ensaio','completo'):
        raise ValueError('Modo de execução inválido.')
    if mode == 'completo':
        if days:
            raise ValueError('Execução completa não admite restrição de dias.')
        selected = dias_do_ano()
    else:
        if not days or len(days) != len(set(days)):
            raise ValueError('Ensaio requer dias explícitos e sem repetição.')
        selected = sorted(days)
        roles = {janela_do_dia(d,config['janelas'])['papel'] for d in selected}
        if not {'train','validation'}.issubset(roles):
            raise ValueError('Ensaio requer pelo menos um dia de treino e um de validação.')
    records = []
    for day in selected:
        window = janela_do_dia(day,config['janelas'])
        records.append(dict(day=day,window=window['chave'],role=window['papel'],files=auditoria.arquivos_do_dia(paths['dados'],day)))
    return records


def planejar(config, paths, mode, days=None):
    records = selecionar(config,paths,mode,days)
    return dict(mode=mode,source=str(paths['dados']),days=len(records),files=sum(len(r['files']) for r in records),
                matrix='B',features=atributos.FEATURES,blocks_percentage=100,
                windows={w['chave']:sum(r['window']==w['chave'] for r in records) for w in config['janelas']},
                phases=PHASES,training_IF='Subconjunto determinístico alvo 500.000; inferência em toda a seleção.',
                training_IF_requested=config['isolation_forest']['training_sample_size'],
                execution_started=False)


def _salvar(path, action):
    document = action()
    escrever_json(path,document)
    return document


def concluir(ctx):
    from . import correlacao,intersecao_mev,figura_intersecoes
    ctx.manifest.update(scientific_status='completed',status='running')
    escrever_json(ctx.path/'manifesto.json',ctx.manifest)
    run,manifest,files,rows=correlacao.prepare(ctx.path)
    result=correlacao.execute(run,manifest,files,rows,ctx.path/'figuras','16GB',8,131072)
    ctx.manifest['correlation_outputs']=result['outputs']
    ctx.manifest['stages']['correlacoes']={k:v for k,v in result.items() if k in ('status','updated_at','elapsed_seconds')}
    result=intersecao_mev.executar(ctx.path,intersecao_mev.planejar(ctx.path,ctx.root),already_locked=True)
    ctx.manifest['intersection_outputs']=result['outputs'].copy()
    ctx.manifest['stages']['intersecoes-mev']={k:v for k,v in result.items()
        if k in ('status','started_at','ended_at','elapsed_seconds')}
    result=figura_intersecoes.gerar(ctx.path)
    ctx.manifest['intersection_outputs'].update(result['outputs'])
    ctx.manifest['stages']['figura-intersecoes-mev']={k:v for k,v in result.items()
        if k in ('status','started_at','ended_at')}
    ctx.finalizar('completed')


def executar(config, paths, mode, days=None, identifier=None, until='relatorios'):
    if until not in PHASES:
        raise ValueError('Etapa final inválida.')
    records = selecionar(config,paths,mode,days)
    if shutil.disk_usage(RAIZ).free < config['recursos']['min_free_gib']*2**30:
        raise ValueError('Espaço livre abaixo da reserva configurada.')
    identifier = identifier or f'{mode}-{uuid.uuid4().hex[:12]}'
    ctx = Execucao(RAIZ,paths['resultados'],identifier,config,paths,[r['day'] for r in records],mode)
    con = None
    try:
        if ctx.manifest.get('scientific_status')=='completed':
            from .comum import sha256
            for path,digest in ctx.manifest['artifacts'].items():
                if not Path(path).is_file() or sha256(Path(path))!=digest:
                    raise ValueError(f'Saída concluída alterada ou ausente: {path}')
            concluir(ctx)
            return ctx.path
        resources = config['recursos']
        con = conectar(ctx.path/'.tmp/duckdb',resources['memory_limit'],resources['threads'])
        if config.get('integridade', {}).get('validar_entradas'):
            from .integridade import conferir
            for r in records:
                check = ctx.path/'integridade-entradas'/f"{r['day']}.json"
                def verificar(r=r, p=check):
                    document = dict(day=r['day'], files=[conferir(f) for f in r['files']])
                    escrever_json(p, document)
                    return dict(rows=sum(f['rows'] for f in document['files']))
                ctx.etapa('verificar-entrada-'+r['day'],r['files'],[check],verificar)
        raw = {}
        for r in records:
            output = ctx.path/'atributos'/r['window']/f"{r['day']}.parquet"
            ctx.etapa('atributos-'+r['day'],r['files'],[output],lambda r=r,p=output:atributos.gerar(con,r['files'],r['day'],p))
            raw.setdefault(r['window'],[]).append(output)
        all_raw = [p for files in raw.values() for p in files]
        coverage = ctx.path/'auditoria-global.json'
        ctx.etapa('auditoria-global',all_raw,[coverage],lambda:_salvar(coverage,lambda:auditoria.auditar_global(con,all_raw)))
        if until == 'atributos':
            ctx.finalizar('partial');return ctx.path
        train_key = next(w['chave'] for w in config['janelas'] if w['papel']=='train')
        validation_key = next(w['chave'] for w in config['janelas'] if w['papel']=='validation')
        scaler = ctx.path/'preprocessamento.json'
        train_raw = raw[train_key]
        ctx.etapa('ajustar-preprocessamento',train_raw,[scaler],lambda:_salvar(scaler,lambda:preprocessamento.ajustar(con,train_raw,'train',ctx.contrato(train_raw))))
        params = ler_json(scaler)
        processed = {}
        for r in records:
            source = ctx.path/'atributos'/r['window']/f"{r['day']}.parquet"
            output = ctx.path/'preprocessado'/r['window']/f"{r['day']}.parquet"
            ctx.etapa('transformar-'+r['day'],[source,scaler],[output],lambda s=source,p=output:preprocessamento.transformar(con,[s],params,p))
            processed.setdefault(r['window'],[]).append(output)
        if until == 'preprocessamento':
            ctx.finalizar('partial');return ctx.path
        # Imports de ML ficam fora do planejamento e da engenharia de atributos.
        from . import autoencoder, isolation_forest
        model_dir = ctx.path/'modelos'
        ae_model, if_model = model_dir/'ae/modelo.pt', model_dir/'if/modelo.joblib'
        fit_inputs = processed[train_key]+[scaler]
        fit_contract = ctx.contrato(fit_inputs)
        ctx.etapa('ajustar-if',fit_inputs,[if_model],lambda:isolation_forest.treinar(con,processed[train_key],config['isolation_forest'],config['seed'],resources['threads'],if_model,fit_contract))
        ctx.etapa('ajustar-ae',fit_inputs,[ae_model,model_dir/'ae/historico.json',model_dir/'ae/checkpoint.pt'],
                  lambda:autoencoder.treinar(con,processed[train_key],config['autoencoder'],config['seed'],model_dir/'ae',fit_contract,ctx.path/'.tmp/ae'))
        if until == 'modelos':
            ctx.finalizar('partial');return ctx.path
        scorers = {'AE':autoencoder.carregar(ae_model,config['autoencoder']['device']),
                   'IF':isolation_forest.carregar(if_model,resources['threads'])}
        models = {'AE':ae_model,'IF':if_model}
        score_files = {m:{} for m in models}
        def inferir(window_keys):
            for m,model in models.items():
                for key in window_keys:
                    if key in score_files[m]:
                        continue
                    for source in processed[key]:
                        output = ctx.path/'escores'/m/key/source.name
                        ctx.etapa(f'escores-{m}-{source.stem}',[source,model],[output],
                                  lambda s=source,p=output,m=m:gravar_escores([s],p,scorers[m],resources['inference_batch_size']))
                        score_files[m].setdefault(key,[]).append(output)
        inferir([validation_key])
        thresholds = {}
        calibration_files = []
        for method, model in models.items():
            output = ctx.path/f'limiares-{method}.json'
            inputs = score_files[method][validation_key]+[model]
            ctx.etapa('calibrar-'+method,inputs,[output],lambda m=method,p=output,model=model,inputs=inputs:_salvar(p,lambda:calibracao.calibrar(con,score_files[m][validation_key],'validation',config['calibracao'],ctx.hash(model),ctx.contrato(inputs))))
            thresholds[method] = ler_json(output)['thresholds']
            calibration_files.append(output)
        if until == 'calibracao':
            ctx.finalizar('partial');return ctx.path
        inferir(list(processed))
        tags_file = ctx.path/'labelcloud.parquet'
        ctx.etapa('labelcloud-normalizar',[paths['labelcloud_csv']],[tags_file],lambda:labelcloud.normalizar(paths['labelcloud_csv'],config['labelcloud'],tags_file))
        labelcloud.criar_tags(con,[tags_file])
        analysis, external, metric_files = {}, {}, []
        for key in raw:
            output = ctx.path/'avaliacao'/f'{key}.json'
            inputs = score_files['AE'][key]+score_files['IF'][key]+calibration_files
            ctx.etapa('avaliacao-'+key,inputs,[output],lambda k=key,p=output:_salvar(p,lambda:avaliacao.resumir(con,score_files['AE'][k],score_files['IF'][k],thresholds['AE'],thresholds['IF'])))
            analysis[key] = ler_json(output)
            metric_files.append(output)
            external[key] = {}
            for method in models:
                output = ctx.path/'labelcloud'/f'{key}-{method}.json'
                inputs = raw[key]+score_files[method][key]+calibration_files+[tags_file]
                ctx.etapa(f'externa-{key}-{method}',inputs,[output],lambda k=key,m=method,p=output:_salvar(p,lambda:avaliacao_externa.avaliar(con,raw[k],score_files[m][k],thresholds[m],config['top_fractions'])))
                external[key][method] = ler_json(output)
                metric_files.append(output)
        drift_file = ctx.path/'drift-B.json'
        all_processed = [p for files in processed.values() for p in files]
        ctx.etapa('drift-B',all_processed,[drift_file],lambda:_salvar(drift_file,lambda:drift.analisar(con,processed,train_key,atributos.FEATURES,config['drift'])))
        for method in models:
            output = ctx.path/f'drift-escores-{method}.json'
            inputs = [p for files in score_files[method].values() for p in files]
            ctx.etapa('drift-escores-'+method,inputs,[output],lambda m=method,p=output:_salvar(p,lambda:drift.analisar(con,score_files[m],train_key,['score'],config['drift'])))
        if until == 'avaliacao':
            ctx.finalizar('partial');return ctx.path
        from . import relatorios
        folder = ctx.path/'figuras'
        outputs = [folder/name for name in relatorios.nomes(analysis)]
        history_file = model_dir/'ae/historico.json'
        ctx.etapa('relatorios',metric_files+[drift_file,history_file],outputs,
                  lambda:relatorios.gerar(folder,analysis,external,ler_json(drift_file),ler_json(history_file)))
        ctx.finalizar('completed')
        con.close();con=None
        concluir(ctx)
        return ctx.path
    except BaseException:
        ctx.finalizar('failed')
        raise
    finally:
        if con is not None:
            con.close()
        ctx.close()

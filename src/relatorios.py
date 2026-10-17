from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .atributos import FEATURES


def nomes(janelas):
    stems=['taxas','jaccard','drift','sobrevivencia','perdas-ae','perdas-ae-detalhe']
    for window in janelas:
        stems += [f'roc-pr-{window}',f'escores-associacao-{window}']
        stems += [f'matriz-{window}-{q}' for q in ('q990','q995','q999')]
    return [f'{stem}.{ext}' for stem in stems for ext in ('pdf','png')]+['roc-pr.pdf','matrizes.pdf']


def salvar(fig, folder, stem):
    fig.savefig(folder/f'{stem}.pdf',bbox_inches='tight')
    fig.savefig(folder/f'{stem}.png',dpi=180,bbox_inches='tight')
    plt.close(fig)


def gerar(folder, analysis, external, drift, history):
    from . import figuras_publicacao as publication
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    windows = list(analysis)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for field,label in [('train_loss','Treino'),('internal_validation_loss','Validação interna')]:
        ax.plot([r['epoch'] for r in history],[r[field] for r in history],label=label)
    ax.set(xlabel='Época',ylabel='MSE de reconstrução')
    ax.legend(); salvar(fig,folder,'perdas-ae')
    fig, ax = plt.subplots(figsize=(11,4),layout='constrained')
    for method in ('AE','IF'):
        ax.plot(windows,[analysis[w]['models'][method]['rates']['q995']['rate']*100 for w in windows],marker='o',label=method)
    ax.set(ylabel='Sinalização em Q99,5 (%)',xlabel='Janela')
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='x',rotation=25)
    ax.legend(); salvar(fig,folder,'taxas')
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    for key in ('q990','q995','q999'):
        ax.plot(windows,[analysis[w]['agreement'][key]['jaccard'] for w in windows],marker='o',label=key)
    ax.set(ylabel='Jaccard',xlabel='Janela',ylim=(0,1))
    ax.tick_params(axis='x',rotation=25)
    ax.legend(); salvar(fig,folder,'jaccard')
    lookup={(r['window'],r['feature']):r['psi'] for r in drift['rows']}
    values=np.array([[lookup[(w,f)] for w in windows] for f in FEATURES])
    fig,ax=plt.subplots(figsize=(11,9),layout='constrained')
    im=ax.imshow(values,aspect='auto',cmap='YlOrRd')
    ax.set_xticks(range(len(windows)),windows,rotation=25,ha='right')
    ax.set_yticks(range(len(FEATURES)),FEATURES)
    fig.colorbar(im,ax=ax,label='PSI em relação ao treino')
    salvar(fig,folder,'drift')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,method in zip(axes,('AE','IF')):
        for w in windows:
            points=[r for r in external[w][method]['curves'] if r['score'] is not None]
            ax.plot([p['score'] for p in points],[max(1e-12,p['survival']) for p in points],label=w)
        ax.set(title=method,xlabel='Escore',ylabel='P(escore ≥ x)',yscale='log')
    axes[-1].legend(fontsize=6)
    salvar(fig,folder,'sobrevivencia')
    with PdfPages(folder/'roc-pr.pdf') as pdf:
        for w in windows:
            fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
            for method in ('AE','IF'):
                r=external[w][method]
                if r['roc_auc'] is None:
                    continue
                points=r['curves']
                axes[0].plot([p['fpr'] for p in points],[p['tpr'] for p in points],label=f"{method}: AUC={r['roc_auc']:.3f}")
                axes[1].plot([p['recall'] for p in points],[p['precision'] for p in points],label=f"{method}: AP={r['average_precision']:.3f}")
            axes[0].plot([0,1],[0,1],'--',color='gray',lw=.7)
            axes[0].set(xlabel='Taxa de falsos positivos da associação',ylabel='Recall da associação')
            axes[1].set(xlabel='Recall da associação',ylabel='Precisão da associação')
            for ax in axes:
                ax.set_xlim(0,1);ax.set_ylim(0,1)
                if ax.lines:
                    ax.legend(fontsize=7)
            fig.suptitle(w+' — avaliação auxiliar Label Cloud')
            fig.savefig(folder/f'roc-pr-{w}.png',dpi=180,bbox_inches='tight')
            pdf.savefig(fig,bbox_inches='tight')
            plt.close(fig)
    data=dict(analysis=analysis,external=external,history=history)
    with PdfPages(folder/'matrizes.pdf') as book:
        for window in windows:
            publication.plot_roc(data,window,folder)
            publication.plot_distribution(data,window,folder)
            for key in publication.THRESHOLDS:
                publication.plot_matrix(data,window,key,folder,book)
    publication.plot_loss(data,folder)
    return dict(files=nomes(windows),curves='Pontos reduzidos apenas para visualização; métricas integradas sobre todos os grupos de escores.')

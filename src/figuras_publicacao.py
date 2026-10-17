"""Figuras vetoriais das avaliações e do treinamento."""
from pathlib import Path
import math

LABELS = {
    'treino': 'Treino: jan-abr/2025',
    'transicao_pectra': 'Transição Pectra: mai/2025',
    'validacao': 'Validação: jun/2025',
    'teste': 'Teste: jul-nov/2025',
    'transicao_fusaka_bpo1': 'Transição Fusaka/BPO1: 1-9/dez/2025',
    'estresse_pos_fusaka_bpo1': 'Estresse temporal: 10-31/dez/2025',
}
THRESHOLDS = {'q990': 'Q99', 'q995': 'Q99,5', 'q999': 'Q99,9'}
BASIC = ['taxas', 'jaccard', 'drift', 'sobrevivencia', 'perdas-ae']
NOTE = 'Avaliação auxiliar por associação de endereços; ausência de associação não confirma benignidade.'


def validate_matrix(result, key, expected_flagged):
    values = result['thresholds'][key]
    for name in ('tn', 'fp', 'fn', 'tp'):
        value = values[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f'Contagem inválida: {key}/{name}')
    tn, fp, fn, tp = (values[k] for k in ('tn', 'fp', 'fn', 'tp'))
    if tn+fp+fn+tp != result['rows'] or fn+tp != result['positives']:
        raise ValueError('Matriz não reconcilia com a população e as associações.')
    if tn+fp != result['unassociated'] or tp+fp != expected_flagged:
        raise ValueError('Matriz não reconcilia com a sinalização por limiar.')
    return [[tn, fp], [fn, tp]]



def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white'})
    return plt


def save(fig, folder, name):
    for ext in ('pdf', 'png'):
        fig.savefig(Path(folder)/f'{name}.{ext}', bbox_inches='tight', dpi=180)


def plot_roc(data, window, folder):
    plt = setup()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), layout='constrained')
    for method, color in [('AE', '#1965a5'), ('IF', '#d97706')]:
        result = data['external'][window][method]
        if result['roc_auc'] is None:
            continue
        points = result['curves']
        axes[0].plot([p['fpr'] for p in points], [p['tpr'] for p in points],
                     color=color, label=f"{method}: AUC={result['roc_auc']:.3f}")
        axes[1].plot([p['recall'] for p in points], [p['precision'] for p in points],
                     color=color, label=f"{method}: AP={result['average_precision']:.3f}")
    prevalence = data['external'][window]['AE']['prevalence']
    axes[0].plot([0,1], [0,1], '--', color='0.6', linewidth=1)
    axes[1].axhline(prevalence, linestyle=':', color='0.45',
                   label=f'Prevalência: {prevalence:.2%}')
    axes[0].set(xlabel='FPR da associação', ylabel='Recall da associação', title='ROC')
    axes[1].set(xlabel='Recall da associação', ylabel='Precisão da associação', title='Precision-Recall')
    for ax in axes:
        ax.set(xlim=(0,1), ylim=(0,1))
        ax.grid(alpha=.16)
        if ax.lines: ax.legend(loc='best', fontsize=9)
    fig.suptitle(LABELS[window])
    fig.supxlabel(NOTE, fontsize=8)
    save(fig, folder, f'roc-pr-{window}')
    plt.close(fig)


def plot_matrix(data, window, key, folder, book=None):
    import numpy as np
    from matplotlib.ticker import PercentFormatter
    plt = setup()
    arrays = {}
    for method in ('AE', 'IF'):
        result = data['external'][window][method]
        flagged = data['analysis'][window]['models'][method]['rates'][key]['flagged']
        arrays[method] = np.array(validate_matrix(result, key, flagged), dtype=float)
    maximum = max(1, *(a.max() for a in arrays.values()))
    fig, axes = plt.subplots(2, 2, figsize=(9.7, 7.8), layout='constrained')
    heat = []
    for column, method in enumerate(('AE', 'IF')):
        raw = arrays[method]
        total = raw.sum(axis=1, keepdims=True)
        norm = np.divide(raw, total, out=np.full_like(raw, np.nan), where=total!=0)
        for row, values in enumerate((raw, norm)):
            ax = axes[row, column]
            vmax = maximum if row==0 else 1
            image = ax.pcolormesh(range(3), range(3), np.ma.masked_invalid(values),
                                  cmap='Blues', vmin=0, vmax=vmax,
                                  edgecolors='white', linewidth=2, rasterized=False)
            if column==1: heat.append(image)
            for i in range(2):
                for j in range(2):
                    v = values[i,j]
                    text = 'n/d' if not math.isfinite(v) else (f'{int(v):,}'.replace(',','.') if row==0 else f'{v:.2%}')
                    ax.text(j+.5, i+.5, text, ha='center', va='center', fontsize=11,
                            color='white' if math.isfinite(v) and v/vmax>.55 else '#172b4d')
            ax.set(xlim=(0,2), ylim=(2,0),
                   xticks=[.5,1.5], yticks=[.5,1.5],
                   xticklabels=['Não sinalizada','Sinalizada'],
                   yticklabels=['Não associado','Associado'],
                   xlabel='Sinalização por escore', ylabel='Associação Label Cloud',
                   title=f'{method} - '+('Contagens' if row==0 else 'Proporção por linha'))
            ax.tick_params(length=0)
    raw_bar = fig.colorbar(heat[0], ax=axes[0,:].tolist(), label='Transações', shrink=.75)
    norm_bar = fig.colorbar(heat[1], ax=axes[1,:].tolist(), label='Proporção por linha',
                           format=PercentFormatter(1), shrink=.75)
    for bar in (raw_bar, norm_bar):
        if bar.solids is not None: bar.solids.set_rasterized(False)
    fig.suptitle(f'{LABELS[window]} - {THRESHOLDS[key]}\nMatrizes de associação auxiliar', fontsize=12)
    fig.supxlabel(NOTE, fontsize=8)
    save(fig, folder, f'matriz-{window}-{key}')
    if book is not None: book.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def plot_distribution(data, window, folder):
    plt = setup()
    fig, axes = plt.subplots(1, 2, figsize=(9.7,4.7), layout='constrained')
    for ax, method in zip(axes, ('AE', 'IF')):
        stats = data['external'][window][method]['score_distributions_by_association']
        stats = sorted(stats, key=lambda s:s['associated'])
        boxes = [dict(label='Associado' if s['associated'] else 'Não associado',
                      q1=s['q25'], med=s['median'], q3=s['q75'],
                      whislo=s['min'], whishi=s['max'], mean=s['mean'], fliers=[])
                 for s in stats]
        ax.bxp(boxes, showfliers=False, showmeans=True,
               meanprops=dict(marker='D',markersize=4),
               medianprops=dict(color='#1965a5',linewidth=1.7))
        if boxes and all(s['whislo']>0 for s in boxes): ax.set_yscale('log')
        ax.set(title=method, ylabel='Escore de anomalia', xlabel='Associação Label Cloud')
        ax.grid(axis='y', alpha=.16)
    fig.suptitle(LABELS[window])
    fig.supxlabel('Caixa: Q25-Q75; linha: mediana; losango: média; hastes: mínimo e máximo.\n'+NOTE, fontsize=8)
    save(fig, folder, f'escores-associacao-{window}')
    plt.close(fig)


def plot_loss(data, folder):
    plt = setup()
    history = data['history']
    best = min(history, key=lambda s:s['internal_validation_loss'])
    fig, axes = plt.subplots(1,2,figsize=(10.6,4.3), layout='constrained')
    for ax, start in zip(axes, (1,20)):
        rows = [s for s in history if s['epoch']>=start]
        for field,label,color in [('train_loss','Treino','#1965a5'),('internal_validation_loss','Validação interna','#d97706')]:
            ax.plot([s['epoch'] for s in rows], [s[field] for s in rows], label=label, color=color)
        ax.axvline(best['epoch'], color='0.5', linestyle=':',label=f"Melhor época: {best['epoch']}")
        ax.set(xlabel='Época', ylabel='MSE de reconstrução', title='Todas as épocas' if start==1 else 'Detalhe a partir da época 20')
        ax.grid(alpha=.16);ax.legend(fontsize=8)
    save(fig,folder,'perdas-ae-detalhe')
    plt.close(fig)

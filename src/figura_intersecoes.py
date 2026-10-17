"""Gera a composição das sinalizações MEV a partir das contagens salvas."""

from pathlib import Path
import importlib.metadata
import tempfile

from .correlacao import Steps,read,sha,signature

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def plot(data,figures):
    figures=Path(figures)
    if data['status'] != 'completed':
        raise ValueError('A contagem das interseções deve estar concluída.')
    windows = [
        ('treino', 'Treino (jan-abr)'),
        ('transicao_pectra', 'Pectra (mai)'),
        ('validacao', 'Validação (jun)'),
        ('teste', 'Teste (jul-nov)'),
        ('transicao_fusaka_bpo1', 'Fusaka/BPO1 (1-9 dez)'),
        ('estresse_pos_fusaka_bpo1', 'Estresse (10-31 dez)'),
    ]
    windows=[(key,label) for key,label in windows if key in data['windows']]
    unknown=set(data['windows'])-{key for key,_ in windows}
    if unknown or not windows:raise ValueError('Recortes desconhecidos ou ausentes na figura MEV.')
    rows = [data['windows'][key]['thresholds']['q995'] for key, _ in windows]
    for row in rows:
        if (any(type(row[k]) is not int or row[k]<0 for k in
                ('AE','IF','AE_only','IF_only','intersection','union'))
            or row['AE_only']+row['intersection']!=row['AE']
            or row['IF_only']+row['intersection']!=row['IF']
            or row['AE_only']+row['intersection']+row['IF_only']!=row['union']):
            raise ValueError('Contagens inconsistentes para a figura MEV.')

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    fig.subplots_adjust(left=.265, right=.985, top=.82, bottom=.19)
    left = [0.] * len(rows)
    for key, label, color, hatch, textcolor in [
        ('AE_only', 'Somente AE', '#0072B2', '', 'white'),
        ('intersection', 'Ambos', '#009E73', '///', 'white'),
        ('IF_only', 'Somente IF', '#E69F00', '', '#202020'),
    ]:
        values = [100 * row[key] / row['union'] if row['union'] else 0. for row in rows]
        ax.barh(range(len(rows)), values, left=left, height=.66,
                label=label, color=color, edgecolor='white', linewidth=.8,
                hatch=hatch, zorder=3)
        for i, value in enumerate(values):
            if value==0:continue
            percent = f'{value:.1f}%'.replace('.', ',')
            if value >= 4:
                ax.text(left[i] + value / 2, i, percent, ha='center',
                        va='center', color=textcolor,
                        fontsize=10.5 if value >= 10 else 8.5,
                        bbox={'facecolor': color, 'edgecolor': 'none', 'pad': .5},
                        zorder=4)
            else:
                ax.annotate(percent, xy=(left[i] + value / 2, i),
                            xytext=(102, i+{'AE_only':-.22,'intersection':0,'IF_only':.22}[key]),
                            ha='left', va='center', fontsize=10,
                            arrowprops={'arrowstyle': '-', 'color': '#555555',
                                        'lw': .65}, zorder=4)
        left = [start + value for start, value in zip(left, values)]
    for i, row in enumerate(rows):
        if not row['union']:
            ax.text(50,i,'Nenhuma transação',ha='center',va='center',color='#666666')
        ax.text(120, i, f"{row['union']:,}".replace(',', '.'),
                ha='center', va='center', fontsize=11)
    ax.text(120, -.78, 'União (n)', ha='center', va='center', fontsize=11)
    ax.set_yticks(range(len(rows)), [label for _, label in windows])
    ax.set_ylim(len(rows) - .45, -.55)
    ax.set_xlim(0, 137)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_xlabel('Participação na união das sinalizações associadas a mev_bot')
    ax.tick_params(axis='y', length=0, pad=8)
    ax.grid(axis='x', color='#dddddd', linewidth=.6, zorder=0)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_bounds(0, 100)
    ax.legend(loc='lower center', bbox_to_anchor=(.44, 1.12), ncol=3,
              frameon=False, handlelength=1.7, columnspacing=1.6)
    figures.mkdir(parents=True,exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix='.intersecoes-',dir=figures) as temporary:
            output=Path(temporary)/'intersecoes-mev-q995'
            fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight', pad_inches=.05)
            fig.savefig(output.with_suffix('.png'), dpi=220, bbox_inches='tight', pad_inches=.05)
            for ext in ('.pdf','.png'):
                output.with_suffix(ext).replace(figures/output.with_suffix(ext).name)
    finally:
        plt.close(fig)


def gerar(run):
    run=Path(run)
    source=run/'avaliacao/intersecao-mev.json'
    source_hash=sha(source)
    identity=dict(input_sha256=source_hash,code_sha256=sha(Path(__file__)),
                  helpers_sha256=sha(Path(__file__).with_name('correlacao.py')),
                  matplotlib=importlib.metadata.version('matplotlib'),threshold='q995')
    folder=run/'.intersecao-mev/figuras'
    steps=Steps(folder,signature(identity),reserve_gib=1)
    outputs=[run/'figuras'/f'intersecoes-mev-q995.{ext}' for ext in ('pdf','png')]
    def action():
        with plt.rc_context():plot(read(source),run/'figuras')
        if sha(source)!=source_hash:raise ValueError('Contagem MEV mudou durante a geração da figura.')
        return dict(threshold='q995',windows=len(read(source)['windows']))
    steps.run('intersecoes-mev-q995',outputs,action)
    return read(folder/'etapas/intersecoes-mev-q995.json')



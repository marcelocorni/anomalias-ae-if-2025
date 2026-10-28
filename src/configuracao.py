from pathlib import Path
import math

from .comum import ler_json,sha256
from .recortes import validar_janelas

RAIZ = Path(__file__).resolve().parents[1]


def validar_hiperparametros_fixos(data):
    selection=data.get('selecao_hiperparametros')
    if selection is None:return
    if (selection.get('metodo')!='parametros_preselecionados_dissertacao'
        or not isinstance(selection.get('referencia'),str)
        or not isinstance(selection.get('sha256'),str)):
        raise ValueError('Referência de hiperparâmetros inválida.')
    reference=(RAIZ/selection['referencia']).resolve()
    if reference.parent!=(RAIZ/'configuracao').resolve() or not reference.is_file():
        raise ValueError('Referência de hiperparâmetros ausente ou fora de configuracao.')
    if sha256(reference)!=selection['sha256']:
        raise ValueError('Hash da referência de hiperparâmetros divergente.')
    winners=ler_json(reference)
    mappings={'AE':('autoencoder',{'hidden_dims':'hidden_dims','latent_dim':'latent_dim',
                                 'learning_rate':'learning_rate','weight_decay':'weight_decay'}),
              'IF':('isolation_forest',{'n_estimators':'n_estimators','max_samples_tree':'max_samples'})}
    for method,(section,fields) in mappings.items():
        parameters=winners.get(method,{}).get('parameters',{})
        if any(source not in parameters or data[section].get(target)!=parameters[source]
               for target,source in fields.items()):
            raise ValueError(f'Hiperparâmetros de {method} divergem dos vencedores da dissertação.')


def carregar(config=None, local=None):
    config = Path(config or RAIZ / 'configuracao/experimento.json').resolve()
    local = Path(local or RAIZ / 'configuracao/local.json').resolve()
    data, locations = ler_json(config), ler_json(local)
    validar_janelas(data['janelas'])
    if data['matriz'] != 'B' or data['percentual_blocos'] != 100 or data['timezone_dados'] != 'UTC':
        raise ValueError('Este pipeline requer matriz B, 100% dos blocos e UTC.')
    if data['calibracao'] != {'q990': .99, 'q995': .995, 'q999': .999}:
        raise ValueError('Quantis de calibração diferentes do desenho aprovado.')
    for section, fields in [('recursos', ['threads', 'inference_batch_size']), ('autoencoder', ['batch_size', 'max_epochs', 'patience', 'internal_validation_modulus']), ('isolation_forest', ['training_sample_size', 'n_estimators', 'max_samples_tree'])]:
        if any(not isinstance(data[section][key], int) or data[section][key] < 1 for key in fields):
            raise ValueError(f'Parâmetros inteiros inválidos em {section}.')
    if data['autoencoder']['internal_validation_modulus'] < 2:
        raise ValueError('A validação interna não pode consumir todo o treino.')
    if not isinstance(data['seed'],int) or not 0 <= data['seed'] < 2**32:
        raise ValueError('Seed fora do intervalo uint32.')
    ae = data['autoencoder']
    if ae['hidden_dims'] != [16,8] or ae['latent_dim'] != 4 or ae['device'] not in ('cpu','cuda'):
        raise ValueError('Arquitetura ou dispositivo AE diferente do contrato B.')
    if any(not math.isfinite(ae[k]) or ae[k] <= 0 for k in ('learning_rate','min_delta')) or not math.isfinite(ae['weight_decay']) or ae['weight_decay'] < 0:
        raise ValueError('Parâmetros de otimização inválidos.')
    if data['drift']['psi_bins'] < 2 or data['drift']['ks_grid_points'] < 2:
        raise ValueError('Grades de drift insuficientes.')
    if not data['top_fractions'] or any(not 0 < f <= 1 for f in data['top_fractions']):
        raise ValueError('Frações top-k inválidas.')
    if data['recursos']['min_free_gib'] < 0:
        raise ValueError('Reserva de disco negativa.')
    validar_hiperparametros_fixos(data)
    paths = {key: (RAIZ / value).resolve() for key, value in locations.items()}
    if set(paths) != {'dados', 'labelcloud_csv', 'resultados'}:
        raise ValueError('Configuração local requer dados, labelcloud_csv e resultados.')
    output = paths['resultados']
    if not output.is_relative_to(RAIZ) or output == RAIZ or output.is_relative_to(RAIZ / 'artefatos'):
        raise ValueError('Resultados devem ficar dentro do novo projeto, fora de artefatos.')
    protected = ['src','scripts','testes','configuracao','documentacao','artigo','.git','.venv']
    if any(output.is_relative_to(RAIZ/name) for name in protected):
        raise ValueError('Resultados não podem ocupar diretórios de código, ambiente ou artigo.')
    if output.is_relative_to(paths['dados']) or paths['dados'].is_relative_to(output):
        raise ValueError('Dados de origem e saídas devem ser disjuntos.')
    if output.is_relative_to(paths['labelcloud_csv'].parent):
        raise ValueError('Saídas não podem ocupar a origem dos rótulos.')
    if not paths['dados'].is_dir() or not paths['labelcloud_csv'].is_file():
        raise FileNotFoundError('Dados ou CSV Label Cloud ausentes.')
    return data, paths

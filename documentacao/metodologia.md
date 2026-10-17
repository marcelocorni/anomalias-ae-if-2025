# Método e contratos

- Dados: Ethereum Mainnet, 2025, timestamps interpretados em UTC. Seleção de 100% dos registros disponíveis; nenhuma amostragem de blocos. Cada partição diária é lida com todos os seus arquivos, preservando o contexto intrabloco.
- Atributos: configuração B, na ordem definida em `src/atributos.py`, com fallback apenas por nulo, sentinelas, ranks com empates e tipo 4 sem indicador próprio. Receipts e contexto do bloco completo caracterizam análise posterior à execução do bloco.
- Identidade: `row_id = (block_number << 32) | transaction_index`, com validação dos limites inteiros e ordenação estável. Não depende da ordem física dos arquivos.
- Integridade: hashes de transação, bloco/posição, datas, tipos, esquema, nulos e finitude são verificados. Falhas interrompem a etapa; não há descarte ou imputação silenciosa. Blocos sem registros e transações finais omitidas precisam de reconciliação externa. O manifesto não afirma cobertura integral da blockchain.
- Pré-processamento: mediana e IQR exatos, ajustados apenas no treino. Nove atributos logarítmicos são normalizados; para taxas máximas, tipos 0/1 são excluídos do ajuste e mantidos em zero. IQR zero usa escala 1. Razões, ranks, indicadores e componentes cíclicos não são normalizados. Saída em double; entrada dos modelos em float32, com verificação de finitude após conversão.

| Janela | Intervalo UTC, fim exclusivo | Uso |
|---|---|---|
| treino | 01/01/2025–01/05/2025 | Scaler e modelos |
| transicao_pectra | 01/05/2025–01/06/2025 | Análise retrospectiva |
| validacao | 01/06/2025–01/07/2025 | Calibração |
| teste | 01/07/2025–01/12/2025 | Avaliação independente |
| transicao_fusaka_bpo1 | 01/12/2025–10/12/2025 | Transição |
| estresse_pos_fusaka_bpo1 | 10/12/2025–01/01/2026 | Estresse temporal |

- AE: `21→16→8→4→8→16→21`, LeakyReLU(0,1), saída linear, MSE e AdamW. Parâmetros na configuração. `row_id % 10 == 0` reserva validação interna dentro do treino. Todos os demais registros participam do ajuste, com matrizes mapeadas em disco e permutação global em CPU (`seed + índice da época`). Checkpoints incluem modelo, melhor estado, otimizador, AMP, RNG e histórico; a retomada ocorre no início da próxima época concluída.
- IF: 200 árvores, até 8.192 amostras por árvore, todas as features, `contamination='auto'`, seed 42. O ajuste usa o subconjunto determinístico `hash(row_id) % N < min(N,500000)`; a quantidade real é registrada e pode diferir do alvo. A ordenação da amostra é estável. A inferência cobre toda a seleção. `max_samples` por árvore e amostragem de treino são políticas distintas.
- Escores: erro quadrático médio para AE; negativo de `score_samples` para IF. Valores maiores indicam maior anomalia. Parquets de escores contêm `row_id` e `score`; sinalizações são reconstruídas com os limiares congelados, sem recalibrar por janela.
- Calibração: quantis exatos Q99, Q99,5 e Q99,9 somente em junho; comparação `>=`. Empates podem elevar a taxa acima do percentual nominal. Q99,5 é a referência principal. Maio usa esses limiares retrospectivamente.
- Drift: PSI com epsilon 1e-6 e KS aproximado sobre a grade de quantis do treino, para B e escores. Não são evidências causais de atualizações do protocolo. Jaccard é calculado nos três limiares; dois conjuntos vazios têm Jaccard 1 por convenção.
- Label Cloud: snapshot de data desconhecida, identificado por hash. Categorias são reconstruídas pela posição histórica; duplicatas são agregadas por endereço antes do join pelo remetente em minúsculas. `mev_bot` é associação positiva fraca. Zero significa ausência dessa associação, sem confirmar benignidade. Não se inferem ataques individuais ou validade temporal do papel do endereço.
- Avaliação externa: ROC-AUC, AP e área PR trapezoidal são calculadas exatamente por grupos de escores empatados, com agregação/ordenação em disco. Apenas os pontos das curvas são reduzidos para plotagem. Sem duas classes, métricas de ranking ficam indisponíveis; denominadores nulos produzem `null`. Top-k usa `ceil(N*f)` e desempate por `row_id`. Nenhum rótulo participa do ajuste ou da calibração.
- Interseções MEV: contagens exatas por recorte em Q99, Q99,5 e Q99,9, restritas às transações associadas a `mev_bot`. São registradas interseção, união e parcelas exclusivas de cada modelo, reconciliadas com a avaliação externa. O gráfico principal usa Q99,5 e normaliza cada união para 100%, indicando também seu total.
- Reprodutibilidade: `uv.lock` fixa o ambiente. Manifestos por execução/etapa registram entradas, saídas e hashes, configurações, código, ambiente, seed, contagens, tempos e memória observada. Escritas são atômicas e concorrência no mesmo identificador é bloqueada. Retomada nunca aceita alterar resultados concluídos.

As figuras são exportadas em PDF e PNG na própria execução. As correlações de Pearson e Spearman são calculadas sobre todos os registros pré-processados do treino, sem amostragem; empates usam postos médios e atributos constantes produzem valores indefinidos.

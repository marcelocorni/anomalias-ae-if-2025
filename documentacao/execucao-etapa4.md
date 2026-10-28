# Execução

Os comandos partem da raiz do projeto. O identificador está em `configuracao/execucao_etapa4.json`; parâmetros em `configuracao/experimento.json` e caminhos locais em `configuracao/local.json`.

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -u ".\scripts\executar_etapa4.py" --executar
```

Use `--verificar` para conferir o ambiente sem iniciar cálculos. O processamento anual abrange os 365 dias de 2025, valida entradas e saídas e salva checkpoints. O mesmo comando retoma etapas compatíveis; uma execução já concluída é conferida sem repetir os cálculos. Alterações científicas exigem outro identificador.

## Parâmetros preselecionados

O perfil `configuracao/execucao_revisao_01.json` usa os vencedores da dissertação registrados em `configuracao/hiperparametros-dissertacao.json`. AE: camadas `[16, 8]`, dimensão latente 4, learning rate 0,0003 e weight decay 0. IF: 100 árvores e 2.048 amostras por árvore. As demais regras do experimento são mantidas; não há nova busca de hiperparâmetros.

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -u ".\scripts\executar_etapa4.py" --perfil "configuracao/execucao_revisao_01.json" --verificar
& ".\.venv\Scripts\python.exe" -X utf8 -u ".\scripts\executar_etapa4.py" --perfil "configuracao/execucao_revisao_01.json" --executar
```

As saídas ficam em `resultados/anual-2025-b-revisao-01/`. O perfil exige a importação conferida das etapas independentes dos modelos antes de iniciar os treinamentos. Chamadas posteriores retomam esse mesmo identificador. A referência dos parâmetros é conferida por SHA-256 e incorporada ao contrato da execução.

## Reaproveitamento da preparação

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -u ".\scripts\reutilizar_preparacao.py" --origem "I:\arquivo-sbrc-2027\anual-2025-b-03" --verificar
& ".\.venv\Scripts\python.exe" -X utf8 -u ".\scripts\reutilizar_preparacao.py" --origem "I:\arquivo-sbrc-2027\anual-2025-b-03" --executar
```

`--verificar` consulta contratos, esquemas, populações e espaço, sem copiar ou calcular hashes de todo o acervo. `--executar` confere os hashes dos dados originais, copia os artefatos e valida integralmente a leitura dos Parquets. São reaproveitados atributos, scaler, matrizes pré-processadas, auditorias, Label Cloud normalizado, drift dos atributos e correlações. A origem permanece intacta; não são criados hardlinks.

A importação cria registros de etapa próprios, com referência aos contratos e hashes da origem. Não copia modelos, escores, limiares ou avaliações. Uma falha preserva as etapas conferidas e bloqueia o pipeline até concluir a importação. Depois de concluída, os cálculos usam as cópias locais, sem depender do HD externo.

## Resultados

- `resultados/<execucao>/atributos/` e `preprocessado/`: atributos diários.
- `modelos/` e `escores/`: modelos, histórico e escores por janela.
- `avaliacao/` e `labelcloud/`: métricas gerais, associação auxiliar e `intersecao-mev.json` com contagens por recorte nos três limiares.
- `figuras/`: gráficos e matrizes em PDF e PNG, além dos coeficientes de correlação em JSON.
- `operacao/`: logs e monitoramento; `.etapas/`, `.correlacao-treino/` e `.intersecao-mev/`: controle de retomada; `.reuso/`: registro da importação.

As matrizes de associação apresentam contagens e proporções nos três limiares congelados. Pearson e Spearman usam os 21 atributos pré-processados de todo o treino. Spearman utiliza postos médios para empates; atributos constantes ficam indefinidos. Nenhum rótulo participa do ajuste dos modelos.

As interseções MEV usam os escores salvos, sem retreinar, com checkpoints diários e reconciliação das contagens por modelo. `figuras/intersecoes-mev-q995.pdf` e `.png` mostram as parcelas exclusivas do AE, compartilhadas e exclusivas do IF, como proporções da união em cada recorte.

Mantenha o terminal aberto. Uma falha encerra a execução sem repetir os cálculos automaticamente. `completed` indica término de todas as etapas, incluindo as figuras.

## Ensaios

Informe pelo menos um dia de treino e um de validação:

```powershell
uv run --frozen python scripts/executar_pipeline.py --modo ensaio --dias 2025-01-01 2025-06-01 2025-07-01 --execucao ensaio-01 --ate atributos --executar
```

`--ate` aceita `atributos`, `preprocessamento`, `modelos`, `calibracao`, `avaliacao` ou `relatorios`. Repita o identificador e os mesmos dias para retomar, alterando apenas a etapa final. `relatorios` inclui todas as figuras, as correlações e as interseções MEV.

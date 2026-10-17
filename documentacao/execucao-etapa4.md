# Execução

Os comandos partem da raiz do projeto. O identificador está em `configuracao/execucao_etapa4.json`; parâmetros em `configuracao/experimento.json` e caminhos locais em `configuracao/local.json`.

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -u ".\scripts\executar_etapa4.py" --executar
```

Use `--verificar` para conferir o ambiente sem iniciar cálculos. O processamento anual abrange os 365 dias de 2025, valida entradas e saídas e salva checkpoints. O mesmo comando retoma etapas compatíveis; uma execução já concluída é conferida sem repetir os cálculos. Alterações científicas exigem outro identificador.

## Resultados

- `resultados/<execucao>/atributos/` e `preprocessado/`: atributos diários.
- `modelos/` e `escores/`: modelos, histórico e escores por janela.
- `avaliacao/` e `labelcloud/`: métricas gerais, associação auxiliar e `intersecao-mev.json` com contagens por recorte nos três limiares.
- `figuras/`: gráficos e matrizes em PDF e PNG, além dos coeficientes de correlação em JSON.
- `operacao/`: logs e monitoramento; `.etapas/`, `.correlacao-treino/` e `.intersecao-mev/`: controle de retomada.

As matrizes de associação apresentam contagens e proporções nos três limiares congelados. Pearson e Spearman usam os 21 atributos pré-processados de todo o treino. Spearman utiliza postos médios para empates; atributos constantes ficam indefinidos. Nenhum rótulo participa do ajuste dos modelos.

As interseções MEV usam os escores salvos, sem retreinar, com checkpoints diários e reconciliação das contagens por modelo. `figuras/intersecoes-mev-q995.pdf` e `.png` mostram as parcelas exclusivas do AE, compartilhadas e exclusivas do IF, como proporções da união em cada recorte.

Mantenha o terminal aberto. Uma falha encerra a execução sem repetir os cálculos automaticamente. `completed` indica término de todas as etapas, incluindo as figuras.

## Ensaios

Informe pelo menos um dia de treino e um de validação:

```powershell
uv run --frozen python scripts/executar_pipeline.py --modo ensaio --dias 2025-01-01 2025-06-01 2025-07-01 --execucao ensaio-01 --ate atributos --executar
```

`--ate` aceita `atributos`, `preprocessamento`, `modelos`, `calibracao`, `avaliacao` ou `relatorios`. Repita o identificador e os mesmos dias para retomar, alterando apenas a etapa final. `relatorios` inclui todas as figuras, as correlações e as interseções MEV.

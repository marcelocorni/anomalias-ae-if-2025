# Ethereum 2025 — SBRC 2027

Comparação temporal de Autoencoder e Isolation Forest com os 21 atributos da configuração B.

## Ambiente

Windows, Python 3.13 e `uv`. O AE utiliza PyTorch com CUDA; as versões estão fixadas em `uv.lock`.

```powershell
uv sync --frozen
Copy-Item configuracao/local.exemplo.json configuracao/local.json
```

Ajuste `configuracao/local.json`. A fonte padrão é `F:/ethereum-2025`; o CSV histórico deve ficar em `dados_auxiliares/labelcloud.csv`, com o hash definido na configuração. Os dados originais são somente leitura.

## Execução

Execute na raiz do projeto:

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -u ".\scripts\executar_etapa4.py" --executar
```

O comando executa os atributos, pré-processamento, modelos, escores, calibração, avaliações, drift, interseções MEV e todas as figuras. Resultados ficam em `resultados/<execucao>/`; PDFs, PNGs e matrizes em `resultados/<execucao>/figuras/`. Pearson e Spearman usam todo o treino, sem amostragem.

Sem `--executar`, o comando apenas verifica o ambiente. A retomada usa o mesmo comando e confere os arquivos concluídos. Em caso de erro, preserve os checkpoints e consulte `resultados/<execucao>/operacao/`.

Para testes e planejamento:

```powershell
uv run --frozen pytest -q
uv run --frozen python scripts/executar_pipeline.py --planejar
```

Ensaios e execução parcial são descritos em [execução](documentacao/execucao-etapa4.md). Consulte também [metodologia](documentacao/metodologia.md). A reconciliação externa dos blocos permanece pendente.

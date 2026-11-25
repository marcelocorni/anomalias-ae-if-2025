"""Reproduz tabelas e gráficos usando os agregados incluídos no pacote LaTeX."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"AE": "#0072B2", "IF": "#D55E00"}
LABELS = {
    "treino": "Treino",
    "transicao_pectra": "Pectra",
    "validacao": "Validação",
    "teste": "Teste",
    "transicao_fusaka_bpo1": "Fusaka/BPO1",
    "estresse_pos_fusaka_bpo1": "Estresse",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def number(value, decimals=2):
    return f"{value:.{decimals}f}".replace(".", ",")


def write_table(path, caption, label, columns, heading, rows):
    content = [
        r"\begin{table}[htbp]",
        r"\centering\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{@{}" + columns + "@{}}",
        r"\toprule",
        heading + r" \\",
        r"\midrule",
    ]
    content.extend(" & ".join(row) + r" \\" for row in rows)
    content.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def tables(data, concentration):
    rows = []
    for window, row in concentration["recortes"].items():
        assert row["associadas"] == data["windows"][window]["external"]["AE"]["positives"]
        rows.append([
            LABELS[window], str(row["remetentes_ativos"]),
            f'{row["associadas"]:,}'.replace(",", "."),
            number(100 * row["top_10"]["fracao_das_associadas"]),
            number(100 * row["top_100"]["fracao_das_associadas"]),
        ])
    write_table(
        ROOT / "tabelas/concentracao-remetentes.tex",
        r"Concentração das transações associadas a \texttt{mev\_bot} por remetente.",
        "tab:concentracao-remetentes", "lrrrr",
        r"Recorte & Ativos & Associadas & Top-10 (\%) & Top-100 (\%)", rows,
    )
    window = data["windows"]["teste"]
    associated = data["mev_intersections"]["windows"]["teste"]["thresholds"]["q995"]
    agreement = window["agreement"]["q995"]
    ae = window["AE_rates"]["q995"]["flagged"]
    forest = window["IF_rates"]["q995"]["flagged"]
    policies = [
        ("Interseção", agreement["intersection"], associated["intersection"]),
        ("IF", forest, associated["IF"]),
        ("AE", ae, associated["AE"]),
        ("União", agreement["union"], associated["union"]),
        ("Somente AE", ae - agreement["intersection"], associated["AE_only"]),
        ("Somente IF", forest - agreement["intersection"], associated["IF_only"]),
    ]
    rows = [[name, f"{n:,}".replace(",", "."), f"{p:,}".replace(",", "."),
             number(100 * p / n)] for name, n, p in policies]
    write_table(
        ROOT / "suplemento/custo-marginal.tex",
        "Volume e associação histórica no teste em Q99,5.",
        "tab:custo-marginal", "lrrr",
        r"Política & Sinalizadas & Associadas & Precision (\%)", rows,
    )
    with (ROOT / "suplemento/recortes-quantis.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["recorte", "modelo", "quantil", "limiar", "sinalizadas", "taxa"])
        for key, record in data["windows"].items():
            for model in COLORS:
                for quantile, value in record[f"{model}_rates"].items():
                    writer.writerow([key, model, quantile, value["threshold"], value["flagged"], value["rate"]])


def precision_recall(data, points):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    formatter = FuncFormatter(lambda value, position: number(value).rstrip("0").rstrip(","))
    windows = [("teste", "(a) Teste (jul-nov)"),
               ("estresse_pos_fusaka_bpo1", "(b) 10-31 dez")]
    for axis, (window, title) in zip(axes, windows):
        inset = axis.inset_axes([0.37, 0.39, 0.60, 0.49])
        for model, color in COLORS.items():
            curve = points[window][model]["curves"]
            metrics = data["windows"][window]["external"][model]
            x = [point["recall"] for point in curve]
            y = [point["precision"] for point in curve]
            axis.plot(x, y, color=color, linewidth=1.3,
                      label=f"{model}: AP={number(metrics['average_precision'], 4)}")
            inset.plot(x, y, color=color, linewidth=1.2)
        prevalence = data["windows"][window]["external"]["AE"]["prevalence"]
        for panel in (axis, inset):
            panel.axhline(prevalence, color="#666666", linestyle="--", linewidth=.8)
            panel.set_xlim(0, 1)
            panel.xaxis.set_major_formatter(formatter)
            panel.yaxis.set_major_formatter(formatter)
            panel.grid(alpha=.15)
        axis.set(title=title, xlabel="Recall da associação", ylabel="Precision da associação", ylim=(0, 1))
        axis.legend(frameon=False, fontsize=8, loc="upper left")
        inset.set_ylim(0, .35)
        inset.set_title("Detalhe até 0,35", fontsize=8)
        inset.tick_params(labelsize=7)
    fig.savefig(ROOT / "imagens/curvas-pr-teste-estresse.pdf", bbox_inches="tight", pad_inches=.05)
    plt.close(fig)


def budget(data):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), layout="constrained")
    for axis, window in zip(axes, ("teste", "estresse_pos_fusaka_bpo1")):
        for model, color in COLORS.items():
            top = data["windows"][window]["external"][model]["top_k"]
            axis.plot(range(len(top)), [100 * row["precision"] for row in top],
                      color=color, marker="o", label=model)
        axis.axhline(100 * data["windows"][window]["external"]["AE"]["prevalence"],
                     color="#666666", linestyle="--", linewidth=.8)
        axis.set(title=LABELS[window], ylabel="Precision da associação (%)",
                 xlabel="Fração do ranking selecionada", ylim=(0, None))
        axis.set_xticks(range(4), ["0,1%", "0,5%", "1%", "5%"])
        axis.legend(frameon=False, fontsize=9)
        axis.grid(axis="y", alpha=.15)
    fig.savefig(ROOT / "imagens/orcamento-investigacao.pdf", bbox_inches="tight", pad_inches=.05)
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    data = read_json(ROOT / "dados-artigo.json")
    concentration = read_json(ROOT / "suplemento/concentracao-remetentes.json")
    assert data["source_signature"] == concentration["assinatura"]
    tables(data, concentration)
    precision_recall(data, read_json(ROOT / "suplemento/curvas-pr.json"))
    budget(data)
    print("Tabelas e gráficos reproduzidos a partir dos agregados do pacote.")


if __name__ == "__main__":
    main()

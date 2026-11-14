"""Build the research article from reviewed aggregates and saved held-out predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import textwrap
import zipfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
OUT = REPORT / "generated"
COLORS = {
    "bert_mini": "#28786b",
    "minilm": "#b47b32",
    "distilbert": "#576ca2",
    "modernbert": "#8d587c",
    "baseline": "#6b6b68",
}
NAMES = {
    "bert_mini": "BERT Mini",
    "minilm": "MiniLM",
    "distilbert": "DistilBERT",
    "modernbert": "ModernBERT",
    "linear_candidate": "Linear candidate",
    "option_prior": "Option prior",
    "uniform": "Uniform",
}
SOURCES: dict[str, str] = {}


def read(path: Path) -> str:
    raw = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return raw.decode()


def save_figure(fig: Figure, name: str) -> None:
    for ext in ("svg", "pdf"):
        fig.savefig(
            OUT / "figures" / f"{name}.{ext}",
            bbox_inches="tight",
            metadata={"Date": None} if ext == "svg" else {"CreationDate": None},
        )
    plt.close(fig)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": "#cccac3",
            "text.color": "#282a28",
            "axes.labelcolor": "#50544f",
            "xtick.color": "#50544f",
            "ytick.color": "#50544f",
            "svg.fonttype": "none",
            "svg.hashsalt": "pocketsolver-report",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def examples() -> list[dict[str, Any]]:
    candidates = [
        ROOT / "artifacts/holdem_meeting5/published_bert_mini/hands.jsonl",
        ROOT / "artifacts/deployment/pocketsolver/assets/hands.jsonl",
    ]
    hand_path = next((p for p in candidates if p.exists()), None)
    if hand_path is None:
        raise FileNotFoundError("Demo hand assets missing; see report/README.md to reproduce them.")
    manifest = json.loads(read(hand_path.with_name("manifest.json")))
    if hashlib.sha256(hand_path.read_bytes()).hexdigest() != manifest["files"]["hands.jsonl"]:
        raise ValueError("Demo hand asset checksum does not match its manifest")
    fold = manifest["fold"]
    protocol = manifest["protocol_sha256"]
    prediction_path = (
        ROOT / f"artifacts/week2_matrix/scenario-cv-{protocol[:16]}/fold-{fold}/oof.jsonl"
    )
    predictions = {r["source_id"]: r for r in map(json.loads, read(prediction_path).splitlines())}
    hands = sorted(map(json.loads, read(hand_path).splitlines()), key=lambda r: r["source_id"])
    selected: dict[str, dict[str, Any]] = {}
    for hand in hands:
        row = predictions[hand["source_id"]]
        p = row["probabilities"]["transformer_linear"]
        baseline = row["probabilities"]["linear_candidate"]
        assert row["fold"] == fold and row["target_index"] == hand["target_index"]
        assert len(p) == len(baseline) == len(hand["candidates"]) > 1
        for probabilities in (p, baseline):
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in probabilities)
            assert math.isclose(sum(probabilities), 1, abs_tol=1e-6)
        guess = max(range(len(p)), key=p.__getitem__)
        target = hand["target_index"]
        if guess == target:
            category = "preflop success" if hand["street"] == "preflop" else "postflop success"
        elif hand["candidates"][guess].split()[0] == hand["candidates"][target].split()[0]:
            category = "wrong amount"
        else:
            category = "wrong action type"
        if category not in selected:
            selected[category] = {
                **hand,
                "category": category,
                "probabilities": p,
                "baseline_probabilities": baseline,
                "prediction_index": guess,
                "fold": fold,
                "protocol_sha256": protocol,
            }
    order = ["postflop success", "preflop success", "wrong action type"]
    assert all(k in selected for k in order), "Missing an example category"
    return [selected[k] for k in order]


def hand_figure(hand: dict[str, Any]) -> None:
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    left.axis("off")
    left.text(
        0,
        1,
        hand["street"].upper() + " / HELD-OUT EXAMPLE",
        va="top",
        fontsize=10,
        color=COLORS["bert_mini"],
    )
    left.text(
        0,
        0.87,
        "\n".join(textwrap.wrap(hand["context"], 49)),
        va="top",
        fontsize=10,
        linespacing=1.6,
    )
    y = np.arange(len(hand["candidates"]))
    right.barh(y, hand["probabilities"], color=COLORS["bert_mini"], height=0.48)
    right.set_yticks(y, hand["candidates"])
    right.invert_yaxis()
    right.set_xlim(0, 1.15)
    right.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    right.set_xlabel("BERT Mini candidate probability")
    for i, value in enumerate(hand["probabilities"]):
        right.text(value + 0.02, i, f"{100 * value:.1f}%", va="center", fontsize=9)
    right.set_title(
        "Reference: " + hand["candidates"][hand["target_index"]], loc="left", fontsize=11, pad=20
    )
    fig.tight_layout(w_pad=2)
    save_figure(fig, "hand")


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.axis("off")
    nodes = [
        (0.11, "State + question\n+ all candidates"),
        (0.39, "Pretrained\nencoder"),
        (0.65, "Mean-pool each\ncandidate span"),
        (0.90, "Shared head\n+ softmax"),
    ]
    for x, label in nodes:
        ax.text(
            x,
            0.6,
            label,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=.7", "fc": "#f0f5f2", "ec": "none"},
        )
    for (x, _), (next_x, _) in zip(nodes, nodes[1:], strict=False):
        ax.annotate(
            "",
            xy=(next_x - 0.11, 0.6),
            xytext=(x + 0.11, 0.6),
            arrowprops={"arrowstyle": "->", "color": "#6b6b68"},
        )
    ax.text(
        0.5,
        0.12,
        "One joint sequence  /  one encoder pass  /  one score per supplied action",
        ha="center",
        color="#50544f",
        fontsize=10,
    )
    save_figure(fig, "architecture")


def model_figure(rows: list[dict[str, Any]], multi: bool = False, mobile: bool = False) -> None:
    fig, axes = plt.subplots(
        2 if mobile else 1,
        1 if mobile else 2,
        figsize=(4.8, 10) if mobile else (10, 5.2),
        sharey=not mobile,
    )
    accuracy = "multi_option_exact_accuracy" if multi else "exact_accuracy"
    for y, row in enumerate(rows):
        color = COLORS[row["backbone"]]
        marker = "D" if row["model"] == "transformer_mlp" else "o"
        for ax, key, scale in [(axes[0], accuracy, 100), (axes[1], "log_loss", 1)]:
            value = row[key] * scale
            ax.scatter(value, y, color=color, marker=marker, s=43, zorder=3)
            ax.text(
                value + (1 if scale == 100 else 0.012), y, f"{value:.2f}", va="center", fontsize=8
            )
    labels = [
        (NAMES.get(r["backbone"]) or "")
        + (" / " + r["model"].split("_")[-1] if r["backbone"] != "baseline" else NAMES[r["model"]])
        for r in rows
    ]
    axes[0].set_yticks(range(len(rows)), labels)
    axes[0].invert_yaxis()
    if mobile:
        axes[1].set_yticks(range(len(rows)), labels)
        axes[1].invert_yaxis()
    axes[0].set_xlim(30, 81)
    axes[1].set_xlim(0.55, 1.12)
    axes[0].set_title("Multi-option accuracy (%)" if multi else "Exact accuracy (%)", loc="left")
    axes[1].set_title("Log loss / all examples", loc="left")
    axes[0].set_xlabel("Higher is better")
    axes[1].set_xlabel("Lower is better")
    for ax in axes:
        ax.grid(axis="x", color="#eeede8")
        ax.tick_params(axis="y", length=0)
    fig.tight_layout(w_pad=3)
    name = "models-multi" if multi else "models"
    save_figure(fig, name + ("-mobile" if mobile else ""))


def street_figure(runs: dict[str, Any], mobile: bool = False) -> None:
    fig, axes = plt.subplots(
        2 if mobile else 1, 1 if mobile else 2, figsize=(4.8, 6.6) if mobile else (10, 3.7)
    )
    streets = ["preflop", "flop", "turn", "river"]
    for backbone in ["bert_mini", "modernbert"]:
        for ax, key, scale in [(axes[0], "exact_accuracy", 100), (axes[1], "log_loss", 1)]:
            ax.plot(
                streets,
                [
                    runs[backbone]["by_street"]["transformer_linear"][s][key] * scale
                    for s in streets
                ],
                "o-",
                color=COLORS[backbone],
                label=NAMES[backbone],
            )
    axes[0].set_ylabel("Exact accuracy (%)")
    axes[1].set_ylabel("Log loss")
    for ax in axes:
        ax.grid(axis="y", color="#eeede8")
        ax.legend(frameon=False)
    fig.tight_layout(w_pad=3)
    save_figure(fig, "streets-mobile" if mobile else "streets")


def leduc_figure(leduc: dict[str, Any], mobile: bool = False) -> None:
    fig, axes = plt.subplots(
        2 if mobile else 1, 1 if mobile else 2, figsize=(4.8, 6.6) if mobile else (10, 3.7)
    )
    keys = ["no_history", "history_4", "history_16", "history_64", "history_all"]
    labels = ["None", "4", "16", "64", "All"]
    for ax, key, scale in [(axes[0], "accuracy", 100), (axes[1], "log_loss", 1)]:
        ax.plot(
            labels,
            [leduc["overall"][k][key] * scale for k in keys],
            "o-",
            color=COLORS["bert_mini"],
        )
        ax.set_xlabel("History window (completed hands)")
        ax.grid(axis="y", color="#eeede8")
    axes[0].set_ylabel("Opponent-action accuracy (%)")
    axes[1].set_ylabel("Log loss")
    fig.tight_layout(w_pad=3)
    save_figure(fig, "leduc-mobile" if mobile else "leduc")


def holdem_figure(rows: list[dict[str, Any]], mobile: bool = False) -> None:
    fig, axes = plt.subplots(
        4 if mobile else 2,
        1 if mobile else 2,
        figsize=(4.8, 12) if mobile else (10, 6.4),
        sharex=True,
    )
    axes = axes.reshape(2, 2)
    for i, model in enumerate(["decision", "forecaster"]):
        for variant, label, color, marker in [
            ("without_history", "No history", "#6b6b68", "o"),
            ("with_history", "With history", COLORS["bert_mini"], "D"),
        ]:
            selected = sorted(
                [
                    r
                    for r in rows
                    if r["model"] == model
                    and r["variant"] == variant
                    and r["validation_scope"] == "expanded"
                ],
                key=lambda r: r["train_examples"],
            )
            assert [r["train_examples"] for r in selected] == [128, 1024, 4096]
            assert all(r["n"] == 1024 for r in selected)
            for j, key, scale in [(0, "accuracy", 100), (1, "log_loss", 1)]:
                axes[i, j].plot(
                    [r["train_examples"] for r in selected],
                    [r[key] * scale for r in selected],
                    marker=marker,
                    label=label,
                    color=color,
                )
        title = "BERT Mini observed-action prediction" if i == 0 else "Public-state forecaster"
        for j in range(2):
            axes[i, j].set_title(title, loc="left", fontsize=10)
            axes[i, j].set_xscale("log", base=2)
            axes[i, j].set_xticks([128, 1024, 4096], ["128", "1,024", "4,096"])
            axes[i, j].grid(axis="y", color="#eeede8")
            axes[i, j].set_ylabel("Accuracy (%)" if j == 0 else "Log loss")
        axes[i, 0].legend(frameon=False, fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("Training decisions (one epoch)")
    fig.tight_layout(h_pad=2, w_pad=3)
    save_figure(fig, "holdem-mobile" if mobile else "holdem")


def generate() -> None:
    for name in ("figures", "tables"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    style()
    rows: list[dict[str, Any]] = [
        dict(row)
        for row in csv.DictReader(read(ROOT / "docs/results/week2-development.csv").splitlines())
    ]
    for row in rows:
        for key in list(row):
            if key not in {"backbone", "model", "protocol_sha256"}:
                row[key] = float(row[key])
    week2 = json.loads(read(ROOT / "docs/results/week2-development.json"))
    for row in rows:
        run = week2["runs"]["bert_mini" if row["backbone"] == "baseline" else row["backbone"]]
        for key in ("exact_accuracy", "log_loss", "multi_option_exact_accuracy"):
            assert math.isclose(row[key], run["models"][row["model"]][key], abs_tol=1e-12)
        assert row["n"] == 10000
    leduc = json.loads(read(ROOT / "docs/results/meeting4-canonical.json"))
    holdem = json.loads(read(ROOT / "docs/results/meeting5-scaling.json"))
    hands = examples()
    hand_figure(hands[0])
    architecture()
    model_figure(rows)
    model_figure(rows, multi=True)
    street_figure(week2["runs"])
    leduc_figure(leduc)
    holdem_figure(holdem["results"])
    model_figure(rows, mobile=True)
    model_figure(rows, multi=True, mobile=True)
    street_figure(week2["runs"], mobile=True)
    leduc_figure(leduc, mobile=True)
    holdem_figure(holdem["results"], mobile=True)
    best = next(
        r for r in rows if r["backbone"] == "modernbert" and r["model"] == "transformer_linear"
    )
    baseline = next(r for r in rows if r["model"] == "linear_candidate")
    distil = next(
        r for r in rows if r["backbone"] == "distilbert" and r["model"] == "transformer_linear"
    )
    values = {
        "best_accuracy": f"{100 * best['exact_accuracy']:.2f}%",
        "best_loss": f"{best['log_loss']:.4f}",
        "baseline_accuracy": f"{100 * baseline['exact_accuracy']:.2f}%",
        "baseline_loss": f"{baseline['log_loss']:.4f}",
        "accuracy_gain": f"{100 * (best['exact_accuracy'] - baseline['exact_accuracy']):.2f}",
        "distil_accuracy": f"{100 * distil['exact_accuracy']:.2f}%",
    }
    for variant in ("without_history", "with_history"):
        for model in ("decision", "forecaster"):
            result = next(
                r
                for r in holdem["results"]
                if r["model"] == model
                and r["variant"] == variant
                and r["train_examples"] == 4096
                and r["validation_scope"] == "expanded"
            )
            prefix = f"holdem_{model}_{variant}"
            values[f"{prefix}_accuracy"] = f"{100 * result['accuracy']:.2f}%"
            values[f"{prefix}_loss"] = f"{result['log_loss']:.6f}"
    for model in ("no_history", "history_selected"):
        result = leduc["overall"][model]
        values[f"leduc_{model}_accuracy"] = f"{100 * result['accuracy']:.2f}%"
        values[f"leduc_{model}_loss"] = f"{result['log_loss']:.4f}"
    payload = {
        "schema": 1,
        "values": values,
        "models": rows,
        "examples": hands,
        "streets": {
            backbone: week2["runs"][backbone]["by_street"]["transformer_linear"]
            for backbone in ("bert_mini", "modernbert")
        },
        "leduc": leduc["overall"],
        "holdem": holdem["results"],
        "scope": "One-seed development comparisons; selected examples are illustrative.",
    }
    (OUT / "report-data.json").write_text(json.dumps(payload, indent=2) + "\n")
    embedded = json.dumps(payload).replace("<", "\\u003c")
    (OUT / "embedded-data.html").write_text(
        '<script id="report-data" type="application/json">' + embedded + "</script>\n"
    )
    with (OUT / "tables/models.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| Model | Head | Exact accuracy | Multi-option accuracy | Log loss |",
        "| :-- | :-- | --: | --: | --: |",
    ]
    for row in rows:
        label = NAMES[row["backbone"]] if row["backbone"] != "baseline" else NAMES[row["model"]]
        head = row["model"].split("_")[-1] if row["backbone"] != "baseline" else "—"
        lines.append(
            f"| {label} | {head} | {100 * row['exact_accuracy']:.2f}% | "
            f"{100 * row['multi_option_exact_accuracy']:.2f}% | {row['log_loss']:.4f} |"
        )
    (OUT / "models-table.qmd").write_text("\n".join(lines) + "\n")
    for source in [
        Path(__file__),
        REPORT / "index.qmd",
        REPORT / "_quarto.yml",
        REPORT / "references.bib",
        *sorted((REPORT / "assets").glob("*")),
        *sorted((REPORT / "filters").glob("*")),
    ]:
        read(source)
    provenance = {
        "sources_sha256": SOURCES,
        "example_selection": (
            "First source ID per category among checksum-verified demo hands; fold-1 OOF join"
        ),
        "outputs_sha256": {
            str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(OUT.rglob("*"))
            if p.is_file() and p.name != "provenance.json"
        },
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        f"Generated {len(rows)} model rows, {len(hands)} verified examples, "
        "and 12 figure pairs (including mobile layouts)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-only", action="store_true")
    parser.add_argument("--quarto", type=Path)
    args = parser.parse_args()
    generate()
    if args.assets_only:
        return
    quarto = str(args.quarto) if args.quarto else shutil.which("quarto")
    if not quarto:
        local = sorted(Path("/tmp/poker-report-tools").glob("quarto-*/bin/quarto"))
        quarto = str(local[-1]) if local else None
    if not quarto:
        parser.error("Quarto is required to render; install it or pass --quarto PATH.")
    for target in ("html", "pdf", "latex"):
        subprocess.run([quarto, "render", "index.qmd", "--to", target], cwd=REPORT, check=True)
    site = REPORT / "_site"
    # Quarto keeps LaTeX image paths relative to the manuscript project.
    shutil.copytree(OUT, site / "generated", dirs_exist_ok=True)
    shutil.copy2(REPORT / "references.bib", site / "references.bib")
    with zipfile.ZipFile(site / "pocketsolver-latex.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(site / "index.tex", "index.tex")
        bundle.write(site / "references.bib", "references.bib")
        for name in ("hand", "architecture", "models", "streets", "leduc", "holdem"):
            image = OUT / "figures" / f"{name}.pdf"
            bundle.write(image, f"generated/figures/{name}.pdf")
        bundle.writestr("README.txt", "Compile index.tex twice with pdflatex from this folder.\n")
    web_archive = site / "pocketsolver-web.zip"
    with zipfile.ZipFile(web_archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for resource in sorted(site.rglob("*")):
            if resource.is_file() and resource != web_archive:
                bundle.write(resource, str(resource.relative_to(site)))
    print(
        f"Article: {site / 'index.html'}\nPaper: {site / 'index.pdf'}\nLaTeX: {site / 'index.tex'}"
    )


if __name__ == "__main__":
    main()

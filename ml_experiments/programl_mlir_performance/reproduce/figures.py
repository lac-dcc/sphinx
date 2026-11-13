"""
Figures of the paper, drawn from the shipped training logs and JSON files.

    python -m reproduce.figures            # all figures
    python -m reproduce.figures rq3 rq8    # a selection

Figures are written as PDF to SPHINX_OUTPUT (default: <repository>/reproduced).
The curves carry the same data as the paper's; colours and layout are plain
matplotlib defaults.
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reproduce import common as c

TASKS = {
    "TrainingTime": "Training time",
    "TestAccuracy": "Test accuracy",
}


def curve(ax, run, label, **kw):
    ax.plot([e["epoch"] for e in run["epochs"]],
            [e["val_tau"] for e in run["epochs"]], label=label, linewidth=1.8, **kw)


def finish(fig, ax, name, xlabel="Epoch", ylabel="Validation Kendall's tau"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    path = c.output_path(name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"written {path}")


def finetuning_figure(folder, source, scratch, scratch_label, name, sub="FT{n}"):
    """Validation tau per epoch for FTE1-8, the source model and the scratch baseline."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    curve(ax, c.load_run(folder, source), "StableHLO (90%)", color="black")
    for n in range(1, 9):
        run = c.load_run(folder, sub.format(n=n))
        assert run["experiment_id"] == n, run["path"]
        curve(ax, run, f"FTE{n}")
    curve(ax, c.load_run(folder, scratch), scratch_label, color="gray", linestyle="--")
    finish(fig, ax, name)


def rq2():
    """Scratch vs fine-tuned at 1%, 5% and 10% of the linalg data."""
    for key, folder in (("TrainingTime", "2.Incremental_Linalg_TrainingTime"),
                        ("TestAccuracy", "4.Incremental_Linalg_TestAccuracy")):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for pct in ("1%", "5%", "10%"):
            curve(ax, c.load_run(folder, pct), f"{pct} scratch", linestyle="--")
            curve(ax, c.load_run(folder, f"{pct}_FT"), f"{pct} fine-tuned")
        finish(fig, ax, f"Section5_TransferEffectiviness_{key}.pdf")


def rq3():
    """All fine-tuning configurations, stablehlo -> linalg."""
    finetuning_figure("1.FineTuning_Batch128_TrainingTime", "TrainingTime_40Epochs_StableHLO",
                      "TrainingTime_100Epochs_Linalg", "Linalg (10%)",
                      "Section5_FineTuning_TrainingTime.pdf")
    finetuning_figure("3.FineTuning_Batch128_TestAccuracy", "TestAccuracy_60Epochs_StableHLO",
                      "TestAccuracy_100Epochs_Linalg", "Linalg (10%)",
                      "Section5_FineTuning_TestAccuracy.pdf")


def rq5():
    """Time per epoch against dataset size, stablehlo and linalg."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sizes = (1, 5, 10, 20, 30, 40, 50, 60)
    for prefix, label in (("HLO", "StableHLO"), ("Linalg", "Linalg")):
        seconds = [c.load_run("5.TimeComparison/TimeComparison", f"{prefix}_{s}")
                   ["epochs"][0]["epoch_seconds"] for s in sizes]
        ax.plot(sizes, seconds, marker="o", label=label)
    finish(fig, ax, "Section5_Time_Comparison.pdf",
           xlabel="Dataset used (%)", ylabel="Time per epoch (s)")


def rq6():
    """All fine-tuning configurations, stablehlo -> scf/memref/arith (and llvm)."""
    for dialect, tt, ta in (("Arith", "6.FineTuning_Batch128_TrainingTime_Arith",
                             "7.FineTuning_Batch128_TestAccuracy_Arith"),
                            ("LLVM", "8.FineTuning_Batch128_TrainingTime_LLVM",
                             "9.FineTuning_Batch128_TestAccuracy_LLVM")):
        finetuning_figure(tt, "TrainingTime_40Epochs_StableHLO",
                          f"TrainingTime_100Epochs_{dialect}", f"{dialect} (10%)",
                          f"Section5_Generalization_TrainingTime_{dialect}.pdf", sub="FTE{n}")
        finetuning_figure(ta, "TestAccuracy_60Epochs_StableHLO",
                          f"TestAccuracy_100Epochs_{dialect}", f"{dialect} (10%)",
                          f"Section5_Generalization_TestAccuracy_{dialect}.pdf", sub="FTE{n}")


def sweep_best():
    """Best-of-three-seeds validation tau per (target, arm, fraction)."""
    best = {}
    for path in ("12.NasBench201_From101/checkpoints/scarcity_sweep_results.json",
                 "15.NasBench201_From101_More/checkpoints/scarcity_sweep_results.json"):
        for r in json.load(open(c.results_path(path))):
            key = (r["target"], r["arm"], r["fraction"])
            best[key] = max(best.get(key, -1.0), r["best_tau"])
    return best


def rq8():
    """NAS-Bench-101 stablehlo -> NAS-Bench-201 linalg, tau against training fraction."""
    best = sweep_best()
    fractions = sorted({k[2] for k in best})
    arms = (("scratch", "Scratch"), ("ft6", "FTE6"), ("ft8", "FTE8"))
    for target, key in (("test_accuracy", "TestAccuracy"), ("training_time", "TrainingTime")):
        fig, ax = plt.subplots(figsize=(6, 4))
        rows = []
        for arm, label in arms:
            values = [best[(target, arm, f)] for f in fractions]
            ax.plot(range(len(fractions)), values, marker="o", label=label)
            rows.append([label] + [f"{v:.3f}" for v in values])
        ax.set_xticks(range(len(fractions)))
        ax.set_xticklabels([f"{int(round(f * 100))}%" for f in fractions])
        finish(fig, ax, f"Section5_Generalization_CrossDataset_{key}.pdf",
               xlabel="Fraction of the NAS-Bench-201 linalg training pool",
               ylabel="Kendall's tau (best of three seeds)")
        print(f"{TASKS[key]}:")
        c.print_table(["Arm"] + [f"{int(round(f * 100))}%" for f in fractions], rows)


FIGURES = {"rq2": rq2, "rq3": rq3, "rq5": rq5, "rq6": rq6, "rq8": rq8}


def main():
    names = sys.argv[1:] or list(FIGURES)
    for name in names:
        if name not in FIGURES:
            sys.exit(f"unknown figure {name!r}; choose from {', '.join(FIGURES)}")
        FIGURES[name]()


if __name__ == "__main__":
    main()

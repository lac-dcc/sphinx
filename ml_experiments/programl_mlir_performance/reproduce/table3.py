"""
Table 3: NAS-Bench-201, both dialect pairs, all predictors.

Reads the JSON files written by pipeline_9010.py and pipeline_9010_2.py. The
values are the best validation Kendall's tau of each run.

    python -m reproduce.table3
"""

import json

from reproduce import common as c

PAIRS = (
    ("stablehlo->linalg",
     "13.NasBench201_StableHLO_to_Linalg/checkpoints/pipeline_9010_results.json"),
    ("arith->llvm",
     "14.NasBench201_Arith_To_LLVM/checkpoints/pipeline_9010_arith_to_llvm_results.json"),
)
TARGETS = ("training_time", "test_accuracy")


def label(arm):
    if arm == "source":
        return "Source (90%)"
    if arm == "scratch":
        return "Target scratch"
    return f"FTE{int(arm[2:])}"


def main():
    table = {}
    for pair, path in PAIRS:
        for r in json.load(open(c.results_path(path))):
            table.setdefault(label(r["arm"]), {})[(pair, r["target"])] = r["best_tau"]

    order = ["Source (90%)", "Target scratch"] + [f"FTE{n}" for n in range(1, 9)]
    header = ["Predictor"] + [f"{p} {t}" for p, _ in PAIRS for t in TARGETS]
    rows = [[name] + [f"{table[name][(p, t)]:.3f}" for p, _ in PAIRS for t in TARGETS]
            for name in order]
    c.print_table(header, rows)


if __name__ == "__main__":
    main()

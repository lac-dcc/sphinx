"""
Table 2: NAS-Bench-101, stablehlo -> linalg, all predictors.

Prints, for both tasks, the epoch time and total time of every run (from the
shipped training logs) and the best validation tau. The test-set metrics of
every column are read from results/test_metrics.json, which is written by
reproduce.eval_checkpoints; run that script to regenerate them from the
shipped checkpoints.

    python -m reproduce.table2
"""

import json
import os

from reproduce import common as c

TASKS = (
    ("Training time", "training_time", "1.FineTuning_Batch128_TrainingTime",
     "TrainingTime_40Epochs_StableHLO", "TrainingTime_100Epochs_Linalg"),
    ("Test accuracy", "test_accuracy", "3.FineTuning_Batch128_TestAccuracy",
     "TestAccuracy_60Epochs_StableHLO", "TestAccuracy_100Epochs_Linalg"),
)


def rmse(t):
    return f"{t['rmse']:.1f}" if t["rmse"] >= 10 else f"{t['rmse']:.3f}"


def main():
    test_path = c.results_path("test_metrics.json")
    test = json.load(open(test_path)) if os.path.exists(test_path) else {}

    for title, target, folder, source, scratch in TASKS:
        runs = [("StableHLO-Source", source)]
        runs += [(f"FTE{n}", f"FT{n}") for n in range(1, 9)]
        runs += [("Linalg-Target", scratch)]

        rows = []
        for label, key in (("Mean Pred.", f"baseline/mean/{target}"),
                           ("Lin. Reg.", f"baseline/linear/{target}"),
                           ("Zero-Shot", f"zero_shot/{target}")):
            t = test.get(key)
            if t:
                rows.append([label, "-", rmse(t), f"{t['r2']:.3f}",
                             f"{t['tau']:.3f}", "-", "-", "-"])
        for label, sub in runs:
            run = c.load_run(folder, sub)
            t = test.get(f"{folder}/{sub}", {})
            rows.append([
                label, sub,
                rmse(t) if t else "-",
                f"{t['r2']:.3f}" if t else "-",
                f"{t['tau']:.3f}" if t else "-",
                f"{c.peak(run):.3f}",
                c.fmt_short(c.mean_epoch_seconds(run)),
                c.fmt_short(c.total_seconds(run)),
            ])
        print(f"\n{title}")
        c.print_table(
            ["Predictor", "Run folder", "Test RMSE", "Test R2", "Test tau",
             "Best val tau", "Epoch time", "Total time"],
            rows,
        )
    if not test:
        print("\nresults/test_metrics.json not found: test-set columns are empty. "
              "Run `python -m reproduce.eval_checkpoints` to produce them.")


if __name__ == "__main__":
    main()

"""
Time-to-performance table: wall-clock time to reach 90%, 95% and 99% of the
peak validation Kendall's tau, from the shipped training logs.

    python -m reproduce.table_time
"""

from reproduce import common as c

THRESHOLDS = (0.90, 0.95, 0.99)

TASKS = (
    ("Training time", "5.TimeComparison/TrainingTime",
     "TrainingTime_40Epochs_StableHLO", "50%"),
    ("Test accuracy", "5.TimeComparison/TestAccuracy",
     "TestAccuracy_60Epochs_StableHLO", "30%"),
)


def first_reaching(run, target):
    for e in run["epochs"]:
        if e["val_tau"] >= target:
            return e
    return None


def main():
    for title, folder, source, scratch in TASKS:
        pre = c.load_run(folder, source)
        base = c.load_run(folder, scratch)
        rows = []

        row = [f"Linalg (Scratch, {scratch})"]
        for pct in THRESHOLDS:
            e = first_reaching(base, c.peak(base) * pct)
            row += ["-", c.fmt_long(e["total_seconds"]), c.fmt_long(e["total_seconds"]),
                    f"{e['val_tau']:.3f}"]
        rows.append(row + [f"{c.peak(base):.3f}"])

        # The folder name and the log header both carry the FTE number.
        for n in (6, 7, 8):
            run = c.load_run(folder, f"FT{n}")
            assert run["experiment_id"] == n, run["path"]
            row = [f"FTE{n} (Transfer, 10%)"]
            for pct in THRESHOLDS:
                p = first_reaching(pre, c.peak(pre) * pct)
                e = first_reaching(run, c.peak(run) * pct)
                row += [c.fmt_long(p["total_seconds"]), c.fmt_long(e["total_seconds"]),
                        c.fmt_long(p["total_seconds"] + e["total_seconds"]),
                        f"{e['val_tau']:.3f}"]
            rows.append(row + [f"{c.peak(run):.3f}"])

        header = ["Predictor"]
        for pct in THRESHOLDS:
            k = int(pct * 100)
            header += [f"{k}% pre-train", f"{k}% linalg", f"{k}% total", f"{k}% val tau"]
        print(f"\n{title}")
        c.print_table(header + ["Peak val tau"], rows)


if __name__ == "__main__":
    main()

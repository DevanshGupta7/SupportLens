"""Train once per seed and summarise. Usage:  python run_seeds.py --bitext --epochs 6"""
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

SEEDS = [1, 2, 3]
extra = sys.argv[1:]  # any train.py options, e.g. --bitext --epochs 6
rows = []
for s in SEEDS:
    out = f"artifacts_seed{s}"
    subprocess.run([sys.executable, "-m", "supportlens.train", "--seed", str(s), "--out", out, *extra], check=True)
    m = json.loads(Path(out, "metrics.json").read_text())
    rows.append((s, m["naive_bayes"]["acc"], m["neural"]["acc"], m["permutation_test_p"]))

print("\nseed   NaiveBayes   Neural    p-value   winner")
for s, nb, nn_, p in rows:
    print(f"{s:<6} {nb:>9.4f} {nn_:>9.4f} {p:>10.4f}   {'neural' if nn_ > nb else 'naive_bayes' if nb > nn_ else 'tie'}")
for name, i in (("NaiveBayes", 1), ("Neural", 2)):
    v = [r[i] for r in rows]
    print(f"{name:<11} mean {st.mean(v):.4f}  std {st.stdev(v):.4f}")
winners = {("neural" if r[2] > r[1] else "naive_bayes") for r in rows}
print("ranking stable across seeds" if len(winners) == 1 else "ranking FLIPS: do not claim one model is better")

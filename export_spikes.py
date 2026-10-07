"""Convert a simulation result (.parquet) to a plain CSV that Blender's Python can read.

Run in the conda env (needs pandas + pyarrow):
    python export_spikes.py [input.parquet] [output.csv]
Output columns: root_id, t   (t in seconds, first trial only, sorted by time)
"""
import sys

import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "Drosophila_brain_model/results/example/flight.parquet"
dst = sys.argv[2] if len(sys.argv) > 2 else "flight_spikes.csv"

df = pd.read_parquet(src)
df = df[df["trial"] == 0]
out = df[["flywire_id", "t"]].rename(columns={"flywire_id": "root_id"}).sort_values("t")
out.to_csv(dst, index=False)

print(f"{len(out)} spikes from {out['root_id'].nunique()} neurons -> {dst}")
print(f"t range: {out['t'].min():.4f} .. {out['t'].max():.4f} s")

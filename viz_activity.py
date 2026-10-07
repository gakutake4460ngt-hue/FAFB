"""Plot the most active neurons from the LIF simulation in 3D, coloured by firing rate.

Input : top_neurons.csv (root_id, rate_hz) exported from the notebook
Output: activity_3d.html (open in a browser)
"""
import sys

import matplotlib
import navis
import pandas as pd
from fafbseg import flywire

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30

flywire.set_default_dataset("public")

df = pd.read_csv("top_neurons.csv").head(N)
ids = df["root_id"].tolist()

try:
    import pyfqmr  # noqa: F401  (backend of navis.simplify_mesh)
    can_simplify = True
except ImportError:
    can_simplify = False
    print("pyfqmr not installed: meshes are not simplified (HTML will be large)")

meshes = []
for i in ids:
    m = flywire.get_mesh_neuron(i)
    if can_simplify:
        m = navis.simplify_mesh(m, F=0.1)  # keeps the HTML light
    meshes.append(m)

norm = matplotlib.colors.Normalize(vmin=df["rate_hz"].min(), vmax=df["rate_hz"].max())
cmap = matplotlib.colormaps["inferno"]
colors = [tuple(int(255 * c) for c in cmap(norm(r))[:3]) for r in df["rate_hz"]]

fig = navis.plot3d(meshes, color=colors, backend="plotly", inline=False)
fig.write_html("activity_3d.html")
print("wrote activity_3d.html")

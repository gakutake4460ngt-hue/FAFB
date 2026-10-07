"""Export the most active neurons as .obj files for Blender.

Input : top_neurons.csv (root_id, rate_hz) exported from the notebook
Output: obj/<root_id>.obj (one file per neuron)

Usage : python export_obj.py [N]
Scale : 1 Blender unit = 10 micrometers (input meshes are in nanometers)
"""
import sys
from pathlib import Path

import navis
import pandas as pd
from fafbseg import flywire

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SCALE = 1e-4  # nm -> units of 10 um

flywire.set_default_dataset("public")

try:
    import pyfqmr  # noqa: F401  (backend of navis.simplify_mesh)
    can_simplify = True
except ImportError:
    can_simplify = False
    print("pyfqmr not installed: meshes are not simplified (files will be large)")

out = Path("obj")
out.mkdir(exist_ok=True)

df = pd.read_csv("top_neurons.csv").head(N)

for i in df["root_id"]:
    m = flywire.get_mesh_neuron(i)
    if can_simplify:
        m = navis.simplify_mesh(m, F=0.1)
    v = m.vertices * SCALE
    with open(out / f"{i}.obj", "w") as f:
        for x, y, z in v:
            f.write(f"v {x:.5f} {y:.5f} {z:.5f}\n")
        for a, b, c in m.faces:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")
    print("wrote", out / f"{i}.obj")

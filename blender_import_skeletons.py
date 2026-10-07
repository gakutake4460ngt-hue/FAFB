"""Import a sample of FlyWire v783 skeletons into Blender, one coloured object per super_class.

Run inside Blender: Scripting tab -> Open this file -> Run Script (Alt+P).
Everything is merged into one edge-only mesh per super_class, so thousands of neurons stay light.
Edit the CONFIG block below if your paths differ.
"""
import csv
import random
from pathlib import Path

import bpy
import numpy as np

# ---- CONFIG -----------------------------------------------------------------
PROJECT = Path.home() / "Desktop/folder_code/projects/lab/FAFB_v783"
SWC_DIR = Path.home() / "Downloads/sk_lod1_783_healed"
ANNOT = PROJECT / "flywire_annotations/supplemental_files/Supplemental_file1_neuron_annotations.tsv"

SCALE = 1e-4          # SWC coordinates are in nm -> 1 Blender unit = 10 um (same as export_obj.py)
SEED = 0

# neurons per super_class (classes with fewer neurons just use all of them)
QUOTA = {
    "central": 800,
    "optic": 600,
    "sensory": 450,
    "visual_projection": 400,
    "ascending": 200,
    "descending": 200,
    "sensory_ascending": 100,
    "visual_centrifugal": 100,
    "motor": 110,
    "endocrine": 80,
}

COLORS = {
    "central": (0.20, 0.55, 1.00),
    "optic": (0.75, 0.25, 0.90),
    "sensory": (0.15, 0.90, 0.45),
    "visual_projection": (1.00, 0.45, 0.10),
    "ascending": (1.00, 0.85, 0.10),
    "descending": (1.00, 0.20, 0.30),
    "sensory_ascending": (0.10, 0.90, 0.90),
    "visual_centrifugal": (0.90, 0.50, 0.75),
    "motor": (1.00, 1.00, 1.00),
    "endocrine": (0.70, 0.70, 0.20),
}
# -----------------------------------------------------------------------------


def read_swc(path):
    """Return (n_nodes x 3 coordinates, edges as pairs of 0-based node indices)."""
    ids, xyz, parents = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            ids.append(int(p[0]))
            xyz.append((float(p[2]), float(p[3]), float(p[4])))
            parents.append(int(p[6]))
    index = {nid: k for k, nid in enumerate(ids)}
    edges = [(index[par], k) for k, par in enumerate(parents) if par in index]
    return np.array(xyz, dtype=np.float32), np.array(edges, dtype=np.int64).reshape(-1, 2)


def pick_neurons():
    random.seed(SEED)
    by_class = {}
    with open(ANNOT) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            sc = r["super_class"]
            if sc in QUOTA and (SWC_DIR / f"{r['root_id']}.swc").exists():
                by_class.setdefault(sc, []).append(r["root_id"])
    return {sc: random.sample(ids, min(QUOTA[sc], len(ids))) for sc, ids in by_class.items()}


def make_material(name, rgb):
    mat = bpy.data.materials.new(f"sk_{name}")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*rgb, 1.0)
    em.inputs["Strength"].default_value = 2.0
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    mat.diffuse_color = (*rgb, 1.0)  # colour shown in Solid viewport mode
    return mat


def build_object(name, ids):
    coords, edges, offset = [], [], 0
    for rid in ids:
        xyz, e = read_swc(SWC_DIR / f"{rid}.swc")
        # same axis convention as Blender's OBJ importer (forward -Z, up Y): (x, y, z) -> (x, -z, y)
        xyz = np.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], axis=1) * SCALE
        coords.append(xyz)
        edges.append(e + offset)
        offset += len(xyz)
    coords = np.concatenate(coords)
    edges = np.concatenate(edges)

    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(coords))
    mesh.vertices.foreach_set("co", coords.ravel())
    mesh.edges.add(len(edges))
    mesh.edges.foreach_set("vertices", edges.ravel())
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    mesh.materials.append(make_material(name, COLORS[name]))
    obj.color = (*COLORS[name], 1.0)
    return obj, coords


def main():
    picked = pick_neurons()
    objs, all_coords = [], []
    for sc, ids in picked.items():
        obj, coords = build_object(sc, ids)
        objs.append(obj)
        all_coords.append(coords)
        print(f"{sc}: {len(ids)} neurons, {len(coords)} nodes")

    # move the whole brain to the origin
    allc = np.concatenate(all_coords)
    center = (allc.min(axis=0) + allc.max(axis=0)) / 2
    for o in objs:
        o.location = (-center[0], -center[1], -center[2])
    print("done:", sum(len(v) for v in picked.values()), "neurons")


main()

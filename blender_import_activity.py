"""Show simulated firing rates on FlyWire v783 skeletons in Blender.

- neurons listed in rates.csv (root_id, rate_hz) -> one object, coloured by firing rate
- a random sample of all other neurons          -> one dim grey object for context

Run inside Blender: Scripting tab -> Open this file -> Run Script (Alt+P).
Use Material Preview (or Rendered) shading to see the colours.
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
RATES = PROJECT / "rates.csv"

SCALE = 1e-4          # SWC coordinates are in nm -> 1 Blender unit = 10 um (same as the other scripts)
N_CONTEXT = 500       # grey background neurons
RADIUS_ACTIVE = 0.04  # tube radius in Blender units (1 unit = 10 um); thicker than real so it is visible at brain scale
RADIUS_CONTEXT = 0.02
RESAMPLE_LENGTH = 0.1  # node spacing along neurites in Blender units (0.1 = 1 um); larger = lighter, 0 = off
VMAX_PERCENTILE = 95  # colour scale tops out at this percentile of the rates (the stimulated cells are far above the rest)
SEED = 0
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


def load_rates():
    rates = {}
    with open(RATES) as f:
        for r in csv.DictReader(f):
            if (SWC_DIR / f"{r['root_id']}.swc").exists():
                rates[r["root_id"]] = float(r["rate_hz"])
    return rates


def pick_context(exclude):
    with open(ANNOT) as f:
        ids = [r["root_id"] for r in csv.DictReader(f, delimiter="\t")
               if r["root_id"] not in exclude and (SWC_DIR / f"{r['root_id']}.swc").exists()]
    random.seed(SEED)
    return random.sample(ids, min(N_CONTEXT, len(ids)))


def build_mesh(name, ids, per_neuron_value=None):
    """Merge neurons into one edge-only mesh; optionally store a per-point 'rate' attribute."""
    coords, edges, values, offset = [], [], [], 0
    for rid in ids:
        xyz, e = read_swc(SWC_DIR / f"{rid}.swc")
        # same axis convention as Blender's OBJ importer (forward -Z, up Y): (x, y, z) -> (x, -z, y)
        xyz = np.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], axis=1) * SCALE
        coords.append(xyz)
        edges.append(e + offset)
        if per_neuron_value is not None:
            values.append(np.full(len(xyz), per_neuron_value[rid], dtype=np.float32))
        offset += len(xyz)
    coords = np.concatenate(coords)
    edges = np.concatenate(edges)

    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(coords))
    mesh.vertices.foreach_set("co", coords.ravel())
    mesh.edges.add(len(edges))
    mesh.edges.foreach_set("vertices", edges.ravel())
    if per_neuron_value is not None:
        attr = mesh.attributes.new(name="rate", type="FLOAT", domain="POINT")
        attr.data.foreach_set("value", np.concatenate(values))
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj, coords


def add_tubes(obj, radius, material):
    """Geometry Nodes: Mesh to Curve -> Curve to Mesh (triangular profile) -> material.

    Edge-only meshes are drawn as thin overlay lines and ignore materials; tubes get shaded and rendered.
    Point attributes (e.g. 'rate') are carried over to the tube surface.
    """
    ng = bpy.data.node_groups.new(f"tubes_{obj.name}", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    n_in = ng.nodes.new("NodeGroupInput")
    n_out = ng.nodes.new("NodeGroupOutput")
    to_curve = ng.nodes.new("GeometryNodeMeshToCurve")
    circle = ng.nodes.new("GeometryNodeCurvePrimitiveCircle")
    circle.inputs["Resolution"].default_value = 3
    circle.inputs["Radius"].default_value = radius
    to_mesh = ng.nodes.new("GeometryNodeCurveToMesh")
    set_mat = ng.nodes.new("GeometryNodeSetMaterial")
    set_mat.inputs["Material"].default_value = material

    ng.links.new(n_in.outputs["Geometry"], to_curve.inputs["Mesh"])
    if RESAMPLE_LENGTH > 0:
        # skeleton nodes are denser than needed: re-place them at even spacing (keeps the shape, drops most points)
        resample = ng.nodes.new("GeometryNodeResampleCurve")
        resample.inputs["Mode"].default_value = "Length"
        resample.inputs["Length"].default_value = RESAMPLE_LENGTH
        ng.links.new(to_curve.outputs["Curve"], resample.inputs["Curve"])
        ng.links.new(resample.outputs["Curve"], to_mesh.inputs["Curve"])
    else:
        ng.links.new(to_curve.outputs["Curve"], to_mesh.inputs["Curve"])
    ng.links.new(circle.outputs["Curve"], to_mesh.inputs["Profile Curve"])
    ng.links.new(to_mesh.outputs["Mesh"], set_mat.inputs["Geometry"])
    ng.links.new(set_mat.outputs["Geometry"], n_out.inputs["Geometry"])

    mod = obj.modifiers.new("tubes", "NODES")
    mod.node_group = ng


def grey_material():
    mat = bpy.data.materials.new("context_grey")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (0.35, 0.40, 0.50, 1.0)
    em.inputs["Strength"].default_value = 0.5
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    mat.diffuse_color = (0.35, 0.40, 0.50, 1.0)
    return mat


def rate_material(vmax):
    """Attribute 'rate' -> Map Range (0..vmax) -> Color Ramp (black-red-orange-yellow) -> Emission."""
    mat = bpy.data.materials.new("firing_rate")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_type = "GEOMETRY"
    attr.attribute_name = "rate"

    rng = nt.nodes.new("ShaderNodeMapRange")
    rng.inputs["From Min"].default_value = 0.0
    rng.inputs["From Max"].default_value = vmax
    rng.clamp = True

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position = 0.0
    cr.elements[0].color = (0.25, 0.0, 0.35, 1.0)
    cr.elements[1].position = 1.0
    cr.elements[1].color = (1.0, 0.95, 0.4, 1.0)
    mid1 = cr.elements.new(0.4)
    mid1.color = (0.85, 0.1, 0.2, 1.0)
    mid2 = cr.elements.new(0.7)
    mid2.color = (1.0, 0.5, 0.05, 1.0)

    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 4.0
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    nt.links.new(attr.outputs["Fac"], rng.inputs["Value"])
    nt.links.new(rng.outputs["Result"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    mat.diffuse_color = (1.0, 0.5, 0.05, 1.0)
    return mat


def main():
    rates = load_rates()
    vmax = float(np.percentile(list(rates.values()), VMAX_PERCENTILE))
    print(f"{len(rates)} active neurons, colour scale 0..{vmax:.1f} Hz")

    ctx_ids = pick_context(set(rates))
    ctx_obj, ctx_coords = build_mesh("context", ctx_ids)
    add_tubes(ctx_obj, RADIUS_CONTEXT, grey_material())

    act_obj, act_coords = build_mesh("active", list(rates), per_neuron_value=rates)
    add_tubes(act_obj, RADIUS_ACTIVE, rate_material(vmax))

    # move the whole brain to the origin
    allc = np.concatenate([ctx_coords, act_coords])
    center = (allc.min(axis=0) + allc.max(axis=0)) / 2
    for o in (ctx_obj, act_obj):
        o.location = (-center[0], -center[1], -center[2])
    print("done")


main()

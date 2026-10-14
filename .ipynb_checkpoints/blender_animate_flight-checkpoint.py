"""Animate simulated firing (flight_spikes.csv) on FlyWire v783 skeletons in Blender.

Each frame shows how recently every neuron fired (bright right after a spike, fading with time constant TAU).
The simulated second is stretched over N_FRAMES frames (slow motion); the camera orbits slowly.

Run inside Blender: Scripting tab -> Open this file -> Run Script (Alt+P).
Then press Space to play, or Render -> Render Animation.
Keep this Blender session open while rendering: the frame handler lives in the session, not in the .blend file.
"""
import csv
import math
import random
from pathlib import Path

import bpy
import numpy as np

# ---- CONFIG -----------------------------------------------------------------
PROJECT = Path.home() / "Desktop/folder_code/projects/lab/FAFB_v783"
SWC_DIR = Path.home() / "Downloads/sk_lod1_783_healed"
ANNOT = PROJECT / "flywire_annotations/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
SPIKES = PROJECT / "flight_spikes.csv"

SIM_T = 1.0           # simulated time in seconds
N_FRAMES = 240        # 240 frames at 24 fps = 10 s video (24x slow motion)
FPS = 24
TAU = 0.02            # how long a spike keeps a neuron lit (seconds of simulated time)

N_ACTIVE = 1000       # most active neurons to show (by spike count)
N_CONTEXT = 300       # grey background neurons
SCALE = 1e-4          # SWC coordinates are in nm -> 1 Blender unit = 10 um
RADIUS_ACTIVE = 0.04
RADIUS_CONTEXT = 0.02
RESAMPLE_LENGTH = 0.1
ORBIT_DEG = 60        # camera sweep over the whole animation
CAM_DIST = 160       # distance of the orbiting camera from the brain centre (the whole brain is ~110 units wide)
SEED = 0
# -----------------------------------------------------------------------------

HANDLER_NAME = "flight_activity_update"


def read_swc(path):
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


def load_activity():
    """Return (neuron ids, activity[frame, neuron] in 0..1) for the N_ACTIVE most active neurons."""
    spikes = {}
    with open(SPIKES) as f:
        for r in csv.DictReader(f):
            spikes.setdefault(r["root_id"], []).append(float(r["t"]))
    have = [(rid, ts) for rid, ts in spikes.items() if (SWC_DIR / f"{rid}.swc").exists()]
    have.sort(key=lambda x: -len(x[1]))
    have = have[:N_ACTIVE]
    ids = [rid for rid, _ in have]

    dt = SIM_T / N_FRAMES
    counts = np.zeros((N_FRAMES, len(ids)), dtype=np.float32)
    for j, (_, ts) in enumerate(have):
        b = np.clip((np.array(ts) / dt).astype(int), 0, N_FRAMES - 1)
        np.add.at(counts[:, j], b, 1.0)

    # leaky trace: decays with TAU, jumps at every spike
    decay = math.exp(-dt / TAU)
    act = np.zeros_like(counts)
    for f in range(N_FRAMES):
        act[f] = (act[f - 1] * decay if f else 0) + counts[f]
    vmax = float(np.percentile(act[act > 0], 99)) if (act > 0).any() else 1.0
    act = np.clip(act / vmax, 0.0, 1.0)
    print(f"{len(ids)} neurons shown; spikes in shown neurons: {int(sum(len(ts) for _, ts in have))}")
    return ids, act


def pick_context(exclude):
    with open(ANNOT) as f:
        ids = [r["root_id"] for r in csv.DictReader(f, delimiter="\t")
               if r["root_id"] not in exclude and (SWC_DIR / f"{r['root_id']}.swc").exists()]
    random.seed(SEED)
    return random.sample(ids, min(N_CONTEXT, len(ids)))


def build_mesh(name, ids):
    """Merge neurons into one edge-only mesh. Returns (object, coords, neuron index of every vertex)."""
    coords, edges, owner, offset = [], [], [], 0
    for k, rid in enumerate(ids):
        xyz, e = read_swc(SWC_DIR / f"{rid}.swc")
        # same axis convention as Blender's OBJ importer (forward -Z, up Y): (x, y, z) -> (x, -z, y)
        xyz = np.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], axis=1) * SCALE
        coords.append(xyz)
        edges.append(e + offset)
        owner.append(np.full(len(xyz), k, dtype=np.int32))
        offset += len(xyz)
    coords, edges, owner = np.concatenate(coords), np.concatenate(edges), np.concatenate(owner)

    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(coords))
    mesh.vertices.foreach_set("co", coords.ravel())
    mesh.edges.add(len(edges))
    mesh.edges.foreach_set("vertices", edges.ravel())
    mesh.attributes.new(name="rate", type="FLOAT", domain="POINT")
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj, coords, owner


def add_tubes(obj, radius, material):
    """Geometry Nodes: Mesh to Curve -> Resample -> Curve to Mesh (triangular profile) -> material."""
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
    obj.modifiers.new("tubes", "NODES").node_group = ng


def grey_material():
    mat = bpy.data.materials.new("context_grey")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (0.20, 0.24, 0.32, 1.0)
    em.inputs["Strength"].default_value = 0.4
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    mat.diffuse_color = (0.20, 0.24, 0.32, 1.0)
    return mat


def glow_material():
    """Attribute 'rate' (0..1) -> Color Ramp (almost black -> purple -> red -> orange -> pale yellow) -> Emission."""
    mat = bpy.data.materials.new("firing_glow")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_type = "GEOMETRY"
    attr.attribute_name = "rate"

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position = 0.0
    cr.elements[0].color = (0.03, 0.02, 0.08, 1.0)
    cr.elements[1].position = 1.0
    cr.elements[1].color = (1.0, 0.97, 0.55, 1.0)
    cr.elements.new(0.25).color = (0.45, 0.05, 0.55, 1.0)
    cr.elements.new(0.55).color = (0.95, 0.12, 0.2, 1.0)
    cr.elements.new(0.8).color = (1.0, 0.55, 0.05, 1.0)

    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 4.0
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(attr.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    mat.diffuse_color = (1.0, 0.5, 0.05, 1.0)
    return mat


def install_handler(mesh, owner, act):
    """Before every frame: write the activity of that frame into the mesh's 'rate' attribute."""
    def flight_activity_update(scene, depsgraph=None):
        f = min(max(scene.frame_current - 1, 0), N_FRAMES - 1)
        mesh.attributes["rate"].data.foreach_set("value", act[f][owner])
        mesh.update()

    flight_activity_update.__name__ = HANDLER_NAME
    for h in list(bpy.app.handlers.frame_change_pre):
        if getattr(h, "__name__", "") == HANDLER_NAME:
            bpy.app.handlers.frame_change_pre.remove(h)
    bpy.app.handlers.frame_change_pre.append(flight_activity_update)
    flight_activity_update(bpy.context.scene)


def setup_scene():
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_start, scene.frame_end = 1, N_FRAMES
    scene.render.resolution_x, scene.render.resolution_y = 1920, 1080
    for eng in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        try:
            scene.render.engine = eng
            break
        except TypeError:
            continue
    scene.render.image_settings.media_type = "VIDEO"  # Blender 5.x: must be set before FFMPEG is offered
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.filepath = str(PROJECT / "flight_")

    scene.view_settings.view_transform = "Standard"  # keeps the glow colours instead of washing them out to white

    # black background
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)

    # camera on an empty at the origin; the empty rotates slowly
    orbit = bpy.data.objects.new("orbit", None)
    bpy.context.collection.objects.link(orbit)
    cam = scene.camera
    if cam is None:
        cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        bpy.context.collection.objects.link(cam)
        scene.camera = cam
    cam.parent = orbit
    cam.location = (0, -CAM_DIST, CAM_DIST * 0.2)
    cam.data.clip_end = 2000
    track = cam.constraints.new("TRACK_TO")
    track.target = orbit
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    orbit.rotation_euler = (0, 0, -math.radians(ORBIT_DEG) / 2)
    orbit.keyframe_insert("rotation_euler", frame=1)
    orbit.rotation_euler = (0, 0, math.radians(ORBIT_DEG) / 2)
    orbit.keyframe_insert("rotation_euler", frame=N_FRAMES)


def main():
    ids, act = load_activity()
    ctx_obj, ctx_coords, _ = build_mesh("context", pick_context(set(ids)))
    add_tubes(ctx_obj, RADIUS_CONTEXT, grey_material())

    act_obj, act_coords, owner = build_mesh("active", ids)
    add_tubes(act_obj, RADIUS_ACTIVE, glow_material())

    # move the whole brain to the origin
    allc = np.concatenate([ctx_coords, act_coords])
    center = (allc.min(axis=0) + allc.max(axis=0)) / 2
    for o in (ctx_obj, act_obj):
        o.location = (-center[0], -center[1], -center[2])

    setup_scene()
    install_handler(act_obj.data, owner, act)
    bpy.context.scene.frame_set(1)
    print("done: press Space to play, or Render -> Render Animation")


main()

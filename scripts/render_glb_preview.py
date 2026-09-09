"""Render headless azimuth previews of a textured Pixal3D GLB.

The orbit azimuths follow the multiview dataset convention: azim000 is the
front view (the face for the bundled example) and the camera circles so the
azimuth names match the reference `assets/mv_images/*/transforms.json`
naming.  When `--transforms` points at that file, the cameras reproduce the
dataset's capture positions exactly (translation from `transform_matrix`,
field of view from `camera_angle_x`), so renders line up with the input
images frame by frame.

Usage:
  blender -b -P scripts/render_glb_preview.py -- <model.glb> <out_prefix> \
      [--transforms transforms.json] [--resolution 768]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("glb", type=Path)
    parser.add_argument("out_prefix", type=Path)
    parser.add_argument("--transforms", type=Path, default=None)
    parser.add_argument("--resolution", type=int, default=768)
    return parser.parse_args(argv)


def look_at(camera, target):
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def main():
    args = parse_args()

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.glb))
    scene = bpy.context.scene
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("world")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 1.0

    mesh_objects = [o for o in scene.objects if o.type == "MESH"]
    mins = Vector((min(o.bound_box[i][j] for o in mesh_objects for i in range(8))
                   for j in range(3)))
    maxs = Vector((max(o.bound_box[i][j] for o in mesh_objects for i in range(8))
                   for j in range(3)))
    center = (mins + maxs) / 2.0
    radius = max((maxs - mins).length, 0.001) * 2.25

    camera_data = bpy.data.cameras.new("cam")
    camera_data.lens = 85
    camera = bpy.data.objects.new("cam", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    sun_data = bpy.data.lights.new("sun", "SUN")
    sun_data.energy = 3.0
    sun = bpy.data.objects.new("sun", sun_data)
    sun.rotation_euler = (math.radians(50), 0, math.radians(30))
    scene.collection.objects.link(sun)

    cameras = []
    if args.transforms:
        transforms = json.loads(args.transforms.read_text())
        camera_data.lens = 18.0 / math.tan(transforms["camera_angle_x"] / 2.0)
        # The capture transforms predate the exporter's H=(-x, +y, -z) flip,
        # which turns the GLB by 180 degrees about the vertical; rotate the
        # camera positions by the same turn so renders line up with the
        # imported GLB (verified against the bundled example views).
        for frame in transforms["frames"]:
            matrix = frame["transform_matrix"]
            position = Vector((-matrix[0][3], -matrix[1][3], matrix[2][3]))
            cameras.append((frame["name"], position))
    else:
        for index in range(4):
            angle = math.radians(90 * index + 180)
            position = center + Vector((radius * math.sin(angle),
                                        -radius * math.cos(angle),
                                        (maxs - mins).length * 0.02))
            cameras.append((f"azim{90 * index:03d}", position))

    for name, position in cameras:
        camera.location = position
        look_at(camera, center)
        scene.render.filepath = f"{args.out_prefix}_{name}.png"
        bpy.ops.render.render(write_still=True)
        print(f"rendered {scene.render.filepath}")


main()

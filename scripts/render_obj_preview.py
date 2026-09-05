"""Render a quick headless preview of a generated Pixal3D OBJ."""

import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def look_at(camera, target):
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("obj", type=Path)
    parser.add_argument("png", type=Path)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.obj_import(filepath=str(args.obj))
    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not objects:
        raise RuntimeError("OBJ did not contain a mesh")

    # Smooth the reconstructed surface without changing the source OBJ.
    for obj in objects:
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
        material = bpy.data.materials.new("Pixal3D preview clay")
        material.diffuse_color = (0.33, 0.48, 0.72, 1.0)
        material.use_nodes = True
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = (0.18, 0.36, 0.68, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.62
        obj.data.materials.append(material)

    points = [obj.matrix_world @ vertex.co for obj in objects for vertex in obj.data.vertices]
    # A tightly capped smoke can contain a few isolated outliers.  Use robust
    # 1st/99th percentile framing so the main reconstruction remains visible.
    def percentile(values, fraction):
        values = sorted(values)
        return values[min(len(values) - 1, int((len(values) - 1) * fraction))]

    minimum = Vector((percentile([point.x for point in points], 0.01),
                      percentile([point.y for point in points], 0.01),
                      percentile([point.z for point in points], 0.01)))
    maximum = Vector((percentile([point.x for point in points], 0.99),
                      percentile([point.y for point in points], 0.99),
                      percentile([point.z for point in points], 0.99)))
    center = (minimum + maximum) * 0.5
    extent = max(maximum.x - minimum.x, maximum.y - minimum.y, maximum.z - minimum.z)
    if extent <= 0:
        extent = 1.0

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(args.png)
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("Pixal3D preview world")
    scene.world.use_nodes = True
    world_nodes = scene.world.node_tree.nodes
    world_nodes["Background"].inputs["Color"].default_value = (0.008, 0.012, 0.025, 1.0)
    world_nodes["Background"].inputs["Strength"].default_value = 0.22

    bpy.ops.object.camera_add(location=(center.x + extent * 1.65, center.y - extent * 1.65, center.z + extent * 0.95))
    camera = bpy.context.object
    camera.data.lens = 58
    camera.data.sensor_width = 36
    look_at(camera, center)
    scene.camera = camera

    def area(name, location, energy, size, color):
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.object
        light.name = name
        light.data.energy = energy
        light.data.shape = "DISK"
        light.data.size = size
        light.data.color = color
        look_at(light, center)

    area("key", (center.x + extent * 1.1, center.y - extent * 1.2, center.z + extent * 1.7), 220, extent * 1.2, (0.78, 0.88, 1.0))
    area("fill", (center.x - extent * 1.0, center.y - extent * 0.5, center.z + extent * 0.35), 90, extent * 1.5, (0.38, 0.55, 1.0))
    area("rim", (center.x - extent * 0.3, center.y + extent * 1.3, center.z + extent * 1.2), 260, extent, (0.52, 0.70, 1.0))

    bpy.ops.mesh.primitive_plane_add(size=extent * 8, location=(center.x, center.y, minimum.z - extent * 0.04))
    ground = bpy.context.object
    ground_mat = bpy.data.materials.new("Preview ground")
    ground_mat.diffuse_color = (0.025, 0.035, 0.06, 1.0)
    ground_mat.use_nodes = True
    ground_bsdf = ground_mat.node_tree.nodes.get("Principled BSDF")
    ground_bsdf.inputs["Base Color"].default_value = (0.018, 0.026, 0.05, 1.0)
    ground_bsdf.inputs["Roughness"].default_value = 0.78
    ground.data.materials.append(ground_mat)

    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.render.filepath = str(args.png)
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()

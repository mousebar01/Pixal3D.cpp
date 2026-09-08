"""Render a quick headless preview of a generated Pixal3D OBJ."""

import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def look_at(camera, target):
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()


def look_at_legacy_obj_front(camera, target):
    # The explicit OBJ compatibility writer uses E=(x, y, z)->(-x, -z, -y).
    # The canonical front camera is on -Y looking toward +Y with +Z image-up.
    # Applying E maps that camera to OBJ +Z looking toward -Z with -Y up.
    # Build the basis explicitly so Blender's default +Y up does not rotate it.
    forward = (Vector(target) - camera.location).normalized()
    up = Vector((0.0, -1.0, 0.0))
    up = (up - forward * up.dot(forward)).normalized()
    right = forward.cross(up).normalized()
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = Matrix((
        (right.x, up.x, -forward.x),
        (right.y, up.y, -forward.y),
        (right.z, up.z, -forward.z),
    )).to_quaternion()


def make_normal_material(name):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    geometry = nodes.new("ShaderNodeNewGeometry")
    multiply = nodes.new("ShaderNodeVectorMath")
    multiply.operation = "MULTIPLY"
    multiply.inputs[1].default_value = (0.5, 0.5, 0.5)
    add = nodes.new("ShaderNodeVectorMath")
    add.operation = "ADD"
    add.inputs[1].default_value = (0.5, 0.5, 0.5)
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs[1].default_value = 1.0
    output = nodes.new("ShaderNodeOutputMaterial")

    links.new(geometry.outputs["Normal"], multiply.inputs[0])
    links.new(multiply.outputs["Vector"], add.inputs[0])
    links.new(add.outputs["Vector"], emission.inputs[0])
    links.new(emission.outputs[0], output.inputs["Surface"])
    return material


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("obj", type=Path)
    parser.add_argument("png", type=Path)
    parser.add_argument("--view", choices=("oblique", "front", "reference-front"), default="oblique")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.obj_import(filepath=str(args.obj))
    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not objects:
        raise RuntimeError("OBJ did not contain a mesh")

    # Smooth the reconstructed surface without changing the source OBJ.  The
    # material is a standard tangent-space normal visualization: RGB=(N+1)/2.
    normal_material = make_normal_material("Pixal3D surface normals")
    for obj in objects:
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
        obj.data.materials.append(normal_material)

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

    camera_distance = extent * 2.25
    if args.view == "front":
        # The official textured Python GLB uses H=(-x, +y, -z), so its
        # reference front is the -Z side looking toward +Z with +Y up.
        camera_location = (center.x, center.y, center.z - camera_distance)
    elif args.view == "reference-front":
        # The explicit OBJ compatibility writer uses E=(-x, -z, -y).  The
        # canonical front camera on -Y therefore becomes OBJ +Z, looking -Z,
        # with OBJ -Y as image-up.
        camera_location = (center.x, center.y, center.z + camera_distance)
    else:
        camera_location = (center.x + camera_distance,
                           center.y - camera_distance,
                           center.z + extent * 1.25)
    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object
    camera.data.lens = 52
    camera.data.sensor_width = 36
    if args.view == "reference-front":
        look_at_legacy_obj_front(camera, center)
    else:
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

    area("key", (center.x + extent * 1.1, center.y - extent * 1.2, center.z - extent * 1.7), 320, extent * 1.2, (0.78, 0.88, 1.0))
    area("fill", (center.x - extent * 1.0, center.y - extent * 0.5, center.z - extent * 0.35), 140, extent * 1.5, (0.48, 0.62, 1.0))
    area("rim", (center.x - extent * 0.3, center.y + extent * 1.3, center.z + extent * 1.2), 260, extent, (0.52, 0.70, 1.0))

    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.filepath = str(args.png)
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()

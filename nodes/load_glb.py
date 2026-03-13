import os
import folder_paths
import trimesh as Trimesh


def _mesh_from_scene(scene_or_mesh):
    if isinstance(scene_or_mesh, Trimesh.Trimesh):
        return scene_or_mesh

    if len(scene_or_mesh.geometry) == 0:
        raise ValueError("GLB contains no geometry.")

    meshes = scene_or_mesh.dump(concatenate=True)
    if isinstance(meshes, Trimesh.Trimesh):
        return meshes
    if len(meshes) == 0:
        raise ValueError("GLB contains no geometry after scene flattening.")
    if len(meshes) == 1:
        return meshes[0]
    return Trimesh.util.concatenate(meshes)


class LoadTrimeshFromGLB:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_path": ("STRING", {"default": "", "tooltip": "Absolute path or input-directory-relative GLB path."}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "load"
    CATEGORY = "3D/Texture Projection"
    DESCRIPTION = "Loads a GLB file and returns a single merged TRIMESH."

    def load(self, glb_path):
        resolved_path = glb_path
        if not os.path.exists(resolved_path):
            resolved_path = os.path.join(folder_paths.get_input_directory(), glb_path)

        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"GLB file not found: {glb_path}")

        scene_or_mesh = Trimesh.load(resolved_path, file_type="glb", force="scene")
        return (_mesh_from_scene(scene_or_mesh),)

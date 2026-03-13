from io import BytesIO

from comfy_api.latest._util.geometry_types import File3D

from ..utils import _as_trimesh, smooth_normals_across_duplicate_positions


class TrimeshToGLB:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "smooth_normals": ("BOOLEAN", {"default": False, "tooltip": "Average normals across duplicate vertex positions before exporting."}),
            }
        }

    RETURN_TYPES = ("FILE_3D_GLB",)
    RETURN_NAMES = ("glb",)
    FUNCTION = "convert"
    CATEGORY = "3D/Texture Projection"
    DESCRIPTION = "Converts a TRIMESH into an in-memory GLB file for native Preview3D and SaveGLB nodes."

    def convert(self, trimesh, smooth_normals):
        trimesh_mesh = _as_trimesh(trimesh)
        if smooth_normals:
            trimesh_mesh = trimesh_mesh.copy()
            trimesh_mesh.vertex_normals = smooth_normals_across_duplicate_positions(trimesh_mesh)
        scene = trimesh_mesh.scene()
        glb_bytes = scene.export(file_type="glb")
        if isinstance(glb_bytes, str):
            glb_bytes = glb_bytes.encode("utf-8")
        return (File3D(BytesIO(glb_bytes), file_format="glb"),)

import numpy as np
import trimesh as Trimesh

from ..utils import (
    _as_trimesh,
    recalculate_outward_vertex_normals,
    smooth_normals_across_duplicate_positions,
)


class SmoothNormals:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "merge_duplicate_vertices": ("BOOLEAN", {"default": True}),
                "recalculate_normals": ("BOOLEAN", {"default": False}),
                "smooth_normals": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "smooth"
    CATEGORY = "3D/Texture Projection"
    DESCRIPTION = "Optionally merge duplicate vertices, recalculate outward-facing normals, and smooth normals across duplicate positions."

    def smooth(
        self,
        trimesh,
        merge_duplicate_vertices,
        recalculate_normals,
        smooth_normals,
    ):
        new_mesh = _as_trimesh(trimesh).copy()

        if bool(merge_duplicate_vertices):
            try:
                new_mesh.merge_vertices()
            except Exception:
                pass

        normals = None
        if bool(recalculate_normals):
            normals = recalculate_outward_vertex_normals(new_mesh)
        elif getattr(new_mesh, "vertices", None) is not None and len(new_mesh.vertices) > 0:
            normals = np.asarray(new_mesh.vertex_normals, dtype=np.float32)

        if bool(smooth_normals):
            normals = smooth_normals_across_duplicate_positions(new_mesh)

        if normals is not None and len(normals) == len(new_mesh.vertices):
            new_mesh.vertex_normals = np.asarray(normals, dtype=np.float32)

        return (new_mesh,)

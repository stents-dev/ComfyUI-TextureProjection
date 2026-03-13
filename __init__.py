import os
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from .nodes.camera import CreateCameraParameters, MultiViewCamera
from .nodes.load_glb import LoadTrimeshFromGLB
from .nodes.project import ProjectImageToMeshUV
from .nodes.render import RenderMeshWithCamera
from .nodes.trimesh_to_glb import TrimeshToGLB


NODE_CLASS_MAPPINGS = {
    "TPCreateCameraParameters": CreateCameraParameters,
    "TPMultiViewCamera": MultiViewCamera,
    "TPLoadMeshFromGLB": LoadTrimeshFromGLB,
    "TPRenderMeshWithCamera": RenderMeshWithCamera,
    "TPProjectImageToMeshUV": ProjectImageToMeshUV,
    "TPTrimeshToGLB": TrimeshToGLB,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "TPCreateCameraParameters": "TextureProjection Camera",
    "TPMultiViewCamera": "TextureProjection MultiView Camera",
    "TPLoadMeshFromGLB": "TextureProjection Load GLB",
    "TPRenderMeshWithCamera": "TextureProjection Render Mesh",
    "TPProjectImageToMeshUV": "TextureProjection Project To UV",
    "TPTrimeshToGLB": "TextureProjection Trimesh To GLB",
}

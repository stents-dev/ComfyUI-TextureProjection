import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import drtk
except Exception as error:
    drtk = None
    _DRTK_IMPORT_ERROR = error
else:
    _DRTK_IMPORT_ERROR = None

from ..utils import (
    _as_trimesh,
    _blur_mask,
    _camera_projection_mode,
    _compute_intrinsics_from_fovy,
    _compute_intrinsics_from_ortho_size,
    _pose_to_w2c,
    _rgb_uint8_to_comfy,
    smooth_normals_across_duplicate_positions,
)


class RenderMeshWithCamera:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "camera": ("CAM_PARAMS",),
                "fit_to_model_bounds": ("BOOLEAN", {"default": False}),
                "smooth_normals": ("BOOLEAN", {"default": True}),
                "smooth_depth": ("BOOLEAN", {"default": False}),
                "transparent_background": ("BOOLEAN", {"default": False}),
                "background_red": ("INT", {"default": 30, "min": 0, "max": 255, "step": 1}),
                "background_green": ("INT", {"default": 30, "min": 0, "max": 255, "step": 1}),
                "background_blue": ("INT", {"default": 30, "min": 0, "max": 255, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "CAM_PARAMS")
    RETURN_NAMES = ("base_color", "depth", "normals", "camera")
    FUNCTION = "render"
    CATEGORY = "3D/Texture Projection"

    @staticmethod
    def _as_bhwc(tensor: torch.Tensor, channels: int) -> torch.Tensor:
        if tensor.ndim != 4:
            raise RuntimeError(f"Expected 4D tensor, got {tuple(tensor.shape)}")
        if tensor.shape[-1] == channels:
            return tensor
        if tensor.shape[1] == channels:
            return tensor.permute(0, 2, 3, 1)
        raise RuntimeError(f"Unexpected tensor shape for {channels} channels: {tuple(tensor.shape)}")

    @staticmethod
    def _as_bchw(tensor: torch.Tensor, channels: int) -> torch.Tensor:
        if tensor.ndim != 4:
            raise RuntimeError(f"Expected 4D tensor, got {tuple(tensor.shape)}")
        if tensor.shape[1] == channels:
            return tensor
        if tensor.shape[-1] == channels:
            return tensor.permute(0, 3, 1, 2)
        raise RuntimeError(f"Unexpected tensor shape for {channels} channels: {tuple(tensor.shape)}")

    @staticmethod
    def _to_rgba_u8(texture, depth: int = 0):
        if texture is None or depth > 4:
            return None

        if isinstance(texture, Image.Image):
            return np.asarray(texture.convert("RGBA"), dtype=np.uint8)

        for attr in ("source", "image", "texture"):
            nested = getattr(texture, attr, None) if hasattr(texture, attr) else None
            if nested is not None and nested is not texture:
                nested_image = RenderMeshWithCamera._to_rgba_u8(nested, depth + 1)
                if nested_image is not None:
                    return nested_image

        try:
            array = np.asarray(texture)
        except Exception:
            return None

        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        if array.ndim != 3:
            return None
        if array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        if array.shape[2] == 3:
            alpha = np.full((array.shape[0], array.shape[1], 1), 255, dtype=array.dtype)
            array = np.concatenate([array, alpha], axis=2)
        elif array.shape[2] > 4:
            array = array[:, :, :4]

        if array.dtype.kind == "f":
            array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
            if float(np.max(array)) <= 1.0 + 1.0e-6:
                array = array * 255.0
            array = np.clip(array, 0.0, 255.0)

        return array.astype(np.uint8, copy=False)

    @staticmethod
    def _material_texture_rgba(visual, material):
        candidates = []
        if visual is not None and hasattr(visual, "image"):
            candidates.append(getattr(visual, "image"))
        if material is not None:
            for attr in ("image", "baseColorTexture"):
                if hasattr(material, attr):
                    candidates.append(getattr(material, attr))

        for texture in candidates:
            image = RenderMeshWithCamera._to_rgba_u8(texture)
            if image is not None:
                return image
        return None

    @staticmethod
    def _material_base_factor(material):
        if material is None or not hasattr(material, "baseColorFactor"):
            return np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        try:
            values = list(material.baseColorFactor)
        except Exception:
            return np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        if len(values) >= 4:
            factor = np.array(values[:4], dtype=np.float32)
        elif len(values) == 3:
            factor = np.array([values[0], values[1], values[2], 1.0], dtype=np.float32)
        else:
            factor = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        if float(np.nanmax(factor)) > 1.5:
            factor = factor / 255.0
        return np.clip(factor, 0.0, 1.0)

    @staticmethod
    def _optimize_axis_shift(values: np.ndarray, depths: np.ndarray, scale: float) -> float:
        left = float(np.min(values))
        right = float(np.max(values))
        if not np.isfinite(left) or not np.isfinite(right):
            return 0.0
        if abs(right - left) < 1.0e-8:
            return 0.5 * (left + right)

        def objective(shift: float) -> float:
            return float(np.max(np.abs(values - shift) * scale - depths))

        for _ in range(64):
            mid_left = left + (right - left) / 3.0
            mid_right = right - (right - left) / 3.0
            if objective(mid_left) <= objective(mid_right):
                right = mid_right
            else:
                left = mid_left
        return 0.5 * (left + right)

    @staticmethod
    def _fit_pose_to_mesh_bounds(
        pose_c2w: np.ndarray,
        vertices_np: np.ndarray,
        projection_mode: str,
        fov_y_deg: float,
        ortho_size: float,
        aspect: float,
    ):
        if vertices_np.size == 0:
            return pose_c2w, ortho_size

        pose_fitted = np.asarray(pose_c2w, dtype=np.float32).copy()
        rotation = pose_fitted[:3, :3].copy()

        vertices_h = np.concatenate(
            [vertices_np.astype(np.float32, copy=False), np.ones((vertices_np.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        world_to_camera = _pose_to_w2c(pose_fitted)
        vertices_camera = (world_to_camera @ vertices_h.T).T[:, :3]

        x = vertices_camera[:, 0].astype(np.float64, copy=False)
        y = vertices_camera[:, 1].astype(np.float64, copy=False)
        z_camera = vertices_camera[:, 2].astype(np.float64, copy=False)
        depths = -z_camera

        if projection_mode == "ortho":
            shift_x = 0.5 * (float(np.min(x)) + float(np.max(x)))
            shift_y = 0.5 * (float(np.min(y)) + float(np.max(y)))
            extent_x = float(np.max(np.abs(x - shift_x))) if x.size else 0.0
            extent_y = float(np.max(np.abs(y - shift_y))) if y.size else 0.0
            fitted_ortho_size = max(1.0e-6, 2.0 * max(extent_y, extent_x / max(aspect, 1.0e-8)) * 1.03)
            delta_camera = np.array([shift_x, shift_y, 0.0], dtype=np.float32)
            pose_fitted[:3, 3] = pose_fitted[:3, 3] + rotation @ delta_camera
            return pose_fitted, fitted_ortho_size

        tan_half = max(math.tan(0.5 * math.radians(float(fov_y_deg))), 1.0e-8)
        scale_x = 1.03 / max(tan_half * max(aspect, 1.0e-8), 1.0e-8)
        scale_y = 1.03 / tan_half

        shift_x = RenderMeshWithCamera._optimize_axis_shift(x, depths, scale_x)
        shift_y = RenderMeshWithCamera._optimize_axis_shift(y, depths, scale_y)

        depth_requirements_x = np.abs(x - shift_x) * scale_x - depths
        depth_requirements_y = np.abs(y - shift_y) * scale_y - depths
        min_depth_requirement = 1.0e-4 - float(np.min(depths))
        shift_z = max(
            0.0,
            float(np.max(depth_requirements_x)) if depth_requirements_x.size else 0.0,
            float(np.max(depth_requirements_y)) if depth_requirements_y.size else 0.0,
            min_depth_requirement,
        )

        shifted_x = x - shift_x
        shifted_y = y - shift_y
        shifted_depths = depths + shift_z
        validation_requirement = np.maximum(
            np.abs(shifted_x) * (1.001 / max(tan_half * max(aspect, 1.0e-8), 1.0e-8)),
            np.abs(shifted_y) * (1.001 / tan_half),
        ) - shifted_depths
        extra_back = max(0.0, float(np.max(validation_requirement)) if validation_requirement.size else 0.0)
        shift_z += extra_back

        delta_camera = np.array([shift_x, shift_y, shift_z], dtype=np.float32)
        pose_fitted[:3, 3] = pose_fitted[:3, 3] + rotation @ delta_camera
        return pose_fitted, ortho_size

    @torch.inference_mode()
    def render(
        self,
        trimesh,
        camera,
        fit_to_model_bounds,
        smooth_normals,
        smooth_depth,
        transparent_background,
        background_red,
        background_green,
        background_blue,
    ):
        if drtk is None:
            raise ImportError(f"drtk import failed: {_DRTK_IMPORT_ERROR}")
        if not torch.cuda.is_available():
            raise RuntimeError("DRTK rendering requires CUDA.")

        trimesh_mesh = _as_trimesh(trimesh)
        pose_c2w = np.asarray(camera["pose_c2w"], dtype=np.float32).copy()
        width = int(camera.get("image_width", camera.get("render_width", 768)))
        height = int(camera.get("image_height", camera.get("render_height", 768)))
        projection_mode = _camera_projection_mode(camera)
        fov_y_deg = float(camera.get("vertical_fov_deg", camera.get("fov_y_deg", 60.0)))
        ortho_size = float(camera.get("orthographic_size", camera.get("ortho_size", 2.0)))

        if not np.isfinite(fov_y_deg):
            fov_y_deg = 60.0
        if not np.isfinite(ortho_size) or ortho_size <= 0.0:
            ortho_size = 2.0

        def build_camera_output():
            camera_output = dict(camera)
            camera_output["pose_c2w"] = pose_c2w.copy()
            camera_output["camera_position"] = pose_c2w[:3, 3].copy()
            camera_output["projection"] = projection_mode
            camera_output["projection_type"] = projection_mode
            camera_output["fov_y_deg"] = fov_y_deg
            camera_output["vertical_fov_deg"] = fov_y_deg
            camera_output["ortho_size"] = ortho_size
            camera_output["orthographic_size"] = ortho_size
            camera_output["render_width"] = width
            camera_output["render_height"] = height
            camera_output["image_width"] = width
            camera_output["image_height"] = height
            if projection_mode == "ortho":
                fx, fy, cx, cy = _compute_intrinsics_from_ortho_size(ortho_size, width, height)
            else:
                fx, fy, cx, cy = _compute_intrinsics_from_fovy(fov_y_deg, width, height)
            camera_output["fx"] = fx
            camera_output["fy"] = fy
            camera_output["cx"] = cx
            camera_output["cy"] = cy
            return camera_output

        vertices_np = np.asarray(trimesh_mesh.vertices, dtype=np.float32)
        faces_np = np.asarray(trimesh_mesh.faces, dtype=np.int32)
        if vertices_np.size == 0 or faces_np.size == 0:
            color_channels = 4 if bool(transparent_background) else 3
            empty_base_color = torch.zeros((1, height, width, color_channels), dtype=torch.float32)
            empty_depth = torch.zeros((1, height, width, 3), dtype=torch.float32)
            empty_normals = torch.zeros((1, height, width, 3), dtype=torch.float32)
            return (empty_base_color, empty_depth, empty_normals, build_camera_output())

        material = getattr(trimesh_mesh.visual, "material", None) if getattr(trimesh_mesh, "visual", None) is not None else None
        texture_rgba_np = self._material_texture_rgba(getattr(trimesh_mesh, "visual", None), material)
        uv_np = None
        if texture_rgba_np is not None and getattr(trimesh_mesh, "visual", None) is not None and hasattr(trimesh_mesh.visual, "uv"):
            uv_candidate = np.asarray(trimesh_mesh.visual.uv, dtype=np.float32)
            if uv_candidate.ndim == 2 and uv_candidate.shape[1] >= 2 and uv_candidate.shape[0] == vertices_np.shape[0]:
                uv_np = uv_candidate[:, :2]

        aspect = float(width) / float(height)
        if bool(fit_to_model_bounds):
            pose_c2w, ortho_size = self._fit_pose_to_mesh_bounds(
                pose_c2w,
                vertices_np,
                projection_mode,
                fov_y_deg,
                ortho_size,
                aspect,
            )

        device = torch.device("cuda")
        vertices = torch.as_tensor(vertices_np, device=device)[None, ...]
        face_indices = torch.as_tensor(faces_np, device=device, dtype=torch.int32)

        if bool(smooth_normals):
            vertex_normals_np = smooth_normals_across_duplicate_positions(trimesh_mesh)
        else:
            vertex_normals_np = np.asarray(trimesh_mesh.vertex_normals, dtype=np.float32)
        vertex_normals = torch.as_tensor(vertex_normals_np, device=device)[None, ...]
        vertex_normals = F.normalize(vertex_normals, dim=-1, eps=1.0e-8)

        ones = torch.ones((1, vertices.shape[1], 1), device=device, dtype=torch.float32)
        vertices_h = torch.cat([vertices, ones], dim=-1)
        world_to_camera = torch.from_numpy(_pose_to_w2c(pose_c2w)).to(device=device, dtype=torch.float32)
        vertices_camera = vertices_h @ world_to_camera.T

        x = vertices_camera[..., 0]
        y = vertices_camera[..., 1]
        z_camera = vertices_camera[..., 2]
        z = (-z_camera).clamp(min=1.0e-8)

        if projection_mode == "ortho":
            half_height = max(0.5 * ortho_size, 1.0e-8)
            half_width = max(half_height * aspect, 1.0e-8)
            x_ndc = x / half_width
            y_ndc = y / half_height
        else:
            tan_half = math.tan(0.5 * math.radians(fov_y_deg))
            x_ndc = (x / z) / (tan_half * aspect)
            y_ndc = (y / z) / tan_half

        x_pixels = (x_ndc + 1.0) * 0.5 * (width - 1)
        y_pixels = (1.0 - y_ndc) * 0.5 * (height - 1)
        raster_vertices = torch.stack([x_pixels, y_pixels, z], dim=-1)

        triangle_index = drtk.rasterize(raster_vertices, face_indices, height=height, width=width)
        render_result = drtk.render(raster_vertices, face_indices, triangle_index)
        barycentric = render_result[1] if isinstance(render_result, tuple) and len(render_result) >= 2 else render_result
        if barycentric is None:
            raise RuntimeError("DRTK render() did not return barycentrics as expected.")

        if triangle_index.ndim == 3:
            triangle_image = triangle_index
        elif triangle_index.ndim == 4 and triangle_index.shape[1] == 1:
            triangle_image = triangle_index[:, 0]
        elif triangle_index.ndim == 4 and triangle_index.shape[-1] == 1:
            triangle_image = triangle_index[..., 0]
        else:
            raise RuntimeError(f"Unexpected triangle index shape: {tuple(triangle_index.shape)}")

        valid = ((triangle_image >= 0) & (triangle_image < int(face_indices.shape[0]))).unsqueeze(1).float()
        valid_bhwc = valid.permute(0, 2, 3, 1)

        base_factor = self._material_base_factor(material)
        base_factor_tensor = torch.as_tensor(base_factor, device=device, dtype=torch.float32).view(1, 1, 1, 4)

        sampled_rgba = None
        if texture_rgba_np is not None and uv_np is not None:
            uv = torch.as_tensor(uv_np, device=device, dtype=torch.float32)[None, ...]
            uv_image = drtk.interpolate(uv, face_indices, triangle_index, barycentric)
            uv_image = self._as_bhwc(uv_image, 2)

            grid_x = uv_image[..., 0] * 2.0 - 1.0
            grid_y = (1.0 - uv_image[..., 1]) * 2.0 - 1.0
            sample_grid = torch.stack([grid_x, grid_y], dim=-1)

            texture_rgba = torch.from_numpy(texture_rgba_np.astype(np.float32) / 255.0).to(device=device)
            texture_rgba = texture_rgba.permute(2, 0, 1).unsqueeze(0).contiguous()
            sampled = F.grid_sample(
                texture_rgba,
                sample_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            sampled_rgba = sampled.permute(0, 2, 3, 1)

        if sampled_rgba is None:
            base_rgb = base_factor_tensor[..., :3].expand(1, height, width, 3)
            base_alpha = base_factor_tensor[..., 3:4].expand(1, height, width, 1)
        else:
            sampled_rgba = sampled_rgba * base_factor_tensor
            base_rgb = sampled_rgba[..., :3]
            base_alpha = sampled_rgba[..., 3:4]

        base_rgb = base_rgb.clamp(0.0, 1.0)
        base_alpha = base_alpha.clamp(0.0, 1.0) * valid_bhwc

        if bool(transparent_background):
            out_rgb = (base_rgb * valid_bhwc).clamp(0.0, 1.0)
            out_base_color = torch.cat([out_rgb, base_alpha], dim=-1)
        else:
            background = torch.tensor(
                [background_red, background_green, background_blue],
                device=device,
                dtype=torch.float32,
            ).view(1, 1, 1, 3) / 255.0
            out_base_color = (base_rgb * base_alpha + background * (1.0 - base_alpha)).clamp(0.0, 1.0)

        depth_image = drtk.interpolate(z.unsqueeze(-1), face_indices, triangle_index, barycentric)
        depth_image = self._as_bchw(depth_image, 1)

        depth_valid = torch.isfinite(depth_image).float() * valid
        if bool(smooth_depth):
            depth_image = (
                _blur_mask(depth_image * depth_valid, radius=2)
                / _blur_mask(depth_valid, radius=2).clamp(min=1.0e-6)
            )
            depth_image = torch.where(depth_valid > 0.0, depth_image, torch.zeros_like(depth_image))

        depth_np = depth_image[0, 0].detach().cpu().numpy().astype(np.float32)
        valid_np = depth_valid[0, 0].detach().cpu().numpy() > 0.5
        valid_np = valid_np & np.isfinite(depth_np) & (depth_np > 0.0)
        if np.any(valid_np):
            depth_min = float(depth_np[valid_np].min())
            depth_max = float(depth_np[valid_np].max())
            depth_range = depth_max - depth_min if depth_max > depth_min else 1.0
            depth_normalized = 1.0 - ((depth_np - depth_min) / depth_range)
            depth_normalized[~valid_np] = 0.0
        else:
            depth_normalized = np.zeros((height, width), dtype=np.float32)

        depth_rgb = np.repeat(depth_normalized[..., None], 3, axis=2)
        out_depth = torch.from_numpy(depth_rgb).unsqueeze(0).clamp(0.0, 1.0)

        normal_image = drtk.interpolate(vertex_normals, face_indices, triangle_index, barycentric)
        normal_image = self._as_bhwc(normal_image, 3)
        normal_image = F.normalize(normal_image, dim=-1, eps=1.0e-8)
        normal_encoded = (normal_image * 0.5 + 0.5).clamp(0.0, 1.0)
        normal_background = torch.tensor([0.5, 0.5, 1.0], device=device, dtype=torch.float32).view(1, 1, 1, 3)
        out_normals = (normal_encoded * valid_bhwc + normal_background * (1.0 - valid_bhwc)).clamp(0.0, 1.0).detach().cpu()

        if not bool(transparent_background):
            color_uint8 = np.clip(out_base_color[0].detach().cpu().numpy() * 255.0 + 0.5, 0, 255).astype(np.uint8)
            out_base_color = _rgb_uint8_to_comfy(color_uint8)
        else:
            out_base_color = out_base_color.detach().cpu()

        return (out_base_color, out_depth, out_normals, build_camera_output())

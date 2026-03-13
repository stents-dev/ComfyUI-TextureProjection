import math

import numpy as np
from PIL import Image
import torch

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
    _dilate_color_by_alpha,
    _dilate_mask,
    _ensure_mask_shape,
    _erode_mask,
    _get_basecolor_image_from_trimesh,
    _pixel_grid_from_uv,
    _pose_to_w2c,
    _set_basecolor_image_on_trimesh,
    _smoothstep,
    _to_comfy_image_tensor,
)


class ProjectImageToMeshUV:
    DEFAULT_OCCLUSION_DILATION = 9
    DEFAULT_OCCLUSION_FADE = 10
    DEFAULT_EDGE_FADE_PIXELS = 24
    DEFAULT_EDGE_FADE_POWER = 1.0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "camera": ("CAM_PARAMS",),
                "image": ("IMAGE",),
                "texture_size": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64}),
                "flip_vertical": ("BOOLEAN", {"default": True}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "backface_cull": ("BOOLEAN", {"default": True}),
                "dilation_pixels": ("INT", {"default": 4, "min": 0, "max": 64, "step": 1}),
                "apply_to_trimesh": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "normal_fade_power": ("FLOAT", {"default": 2.2, "min": 0.1, "max": 8.0, "step": 0.01}),
                "normal_fade_min": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "projection_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("trimesh", "texture_map", "projected_rgb", "projected_alpha")
    FUNCTION = "project"
    CATEGORY = "3D/Texture Projection"

    def project(
        self,
        trimesh,
        camera,
        image,
        texture_size,
        flip_vertical,
        opacity,
        backface_cull,
        dilation_pixels,
        apply_to_trimesh,
        normal_fade_power=2.2,
        normal_fade_min=0.1,
        projection_mask=None,
    ):
        if drtk is None:
            raise ImportError(f"drtk import failed: {_DRTK_IMPORT_ERROR}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. DRTK GPU projection requires CUDA.")

        trimesh_for_projection = _as_trimesh(trimesh)
        if trimesh_for_projection.visual is None or not hasattr(trimesh_for_projection.visual, "uv") or trimesh_for_projection.visual.uv is None:
            raise ValueError("Mesh has no UVs. Texture projection requires existing UVs.")

        device = torch.device("cuda")
        vertices = torch.as_tensor(np.asarray(trimesh_for_projection.vertices, np.float32), device=device)[None, ...]
        faces = torch.as_tensor(np.asarray(trimesh_for_projection.faces, np.int32), device=device)
        uv = torch.as_tensor(np.asarray(trimesh_for_projection.visual.uv, np.float32), device=device)[None, ...]
        if flip_vertical:
            uv = torch.stack([uv[..., 0], 1.0 - uv[..., 1]], dim=-1)

        vertex_normals_np = np.asarray(trimesh_for_projection.vertex_normals, np.float32)
        vertex_normals = torch.as_tensor(vertex_normals_np, device=device)[None, ...]

        size = int(texture_size)
        uv_pixels = torch.empty((1, uv.shape[1], 3), device=device, dtype=torch.float32)
        uv_pixels[..., 0] = uv[..., 0] * (size - 1)
        uv_pixels[..., 1] = uv[..., 1] * (size - 1)
        uv_pixels[..., 2] = 1.0

        uv_triangle_index = drtk.rasterize(uv_pixels, faces, height=size, width=size)
        uv_render = drtk.render(uv_pixels, faces, uv_triangle_index)
        barycentric = uv_render[1] if isinstance(uv_render, tuple) and len(uv_render) >= 2 else uv_render
        if barycentric is None:
            raise RuntimeError("DRTK render() did not return barycentrics as expected.")

        uv_mask = (uv_triangle_index != -1).float().unsqueeze(1)

        world_position = drtk.interpolate(vertices, faces, uv_triangle_index, barycentric)
        if world_position.ndim == 4 and world_position.shape[1] == 3:
            world_position = world_position.permute(0, 2, 3, 1)
        elif world_position.ndim != 4 or world_position.shape[-1] != 3:
            raise RuntimeError(f"Unexpected world position shape: {tuple(world_position.shape)}")

        interpolated_normals = drtk.interpolate(vertex_normals, faces, uv_triangle_index, barycentric)
        if interpolated_normals.ndim == 4 and interpolated_normals.shape[1] == 3:
            interpolated_normals = interpolated_normals.permute(0, 2, 3, 1)
        elif interpolated_normals.ndim != 4 or interpolated_normals.shape[-1] != 3:
            raise RuntimeError(f"Unexpected interpolated normal shape: {tuple(interpolated_normals.shape)}")

        camera_position = torch.as_tensor(
            np.asarray(camera["pose_c2w"][:3, 3], np.float32),
            device=device,
        ).view(1, 1, 1, 3)

        vertex_normals_view = torch.nn.functional.normalize(vertex_normals, dim=-1)
        vertex_view_directions = torch.nn.functional.normalize(
            camera_position[:, :1, :1, :].squeeze(2) - vertices,
            dim=-1,
        )
        normal_dot_vertex = (vertex_normals_view * vertex_view_directions).sum(dim=-1, keepdim=True)
        normal_dot = drtk.interpolate(normal_dot_vertex, faces, uv_triangle_index, barycentric)
        if normal_dot.ndim == 4 and normal_dot.shape[1] == 1:
            normal_dot = normal_dot.permute(0, 2, 3, 1)
        elif normal_dot.ndim != 4 or normal_dot.shape[-1] != 1:
            raise RuntimeError(f"Unexpected interpolated dot shape: {tuple(normal_dot.shape)}")

        if backface_cull:
            facing = _smoothstep(
                torch.full_like(normal_dot, -0.02),
                torch.full_like(normal_dot, 0.02),
                normal_dot,
            )
            uv_mask = uv_mask * facing.permute(0, 3, 1, 2)

        image_tensor = image
        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError("image must be a ComfyUI IMAGE tensor.")

        image_tensor = image_tensor.to(device=device, dtype=torch.float32, non_blocking=True)
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if image_tensor.ndim != 4:
            raise ValueError(f"Expected image as [B,H,W,C] or [H,W,C], got {tuple(image_tensor.shape)}")
        if image_tensor.shape[-1] < 3:
            raise ValueError("Projected image must have at least 3 channels.")

        image_rgb = image_tensor[..., :3]
        image_alpha = image_tensor[..., 3:4] if image_tensor.shape[-1] >= 4 else torch.ones_like(image_tensor[..., :1])

        if projection_mask is not None:
            mask_tensor = projection_mask
            if not isinstance(mask_tensor, torch.Tensor):
                mask_tensor = torch.as_tensor(mask_tensor)
            mask_tensor = mask_tensor.to(device=device, dtype=torch.float32, non_blocking=True)

            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(-1)
            elif mask_tensor.ndim == 3:
                mask_tensor = mask_tensor.unsqueeze(-1)
            elif mask_tensor.ndim == 4:
                if mask_tensor.shape[-1] == 1:
                    pass
                elif mask_tensor.shape[1] == 1:
                    mask_tensor = mask_tensor.permute(0, 2, 3, 1)
                else:
                    mask_tensor = mask_tensor[..., :1]
            else:
                raise ValueError(f"projection_mask has unsupported shape: {tuple(mask_tensor.shape)}")

            if mask_tensor.shape[0] != image_alpha.shape[0]:
                if mask_tensor.shape[0] == 1:
                    mask_tensor = mask_tensor.expand(image_alpha.shape[0], -1, -1, -1)
                elif image_alpha.shape[0] == 1:
                    image_alpha = image_alpha.expand(mask_tensor.shape[0], -1, -1, -1)
                    image_rgb = image_rgb.expand(mask_tensor.shape[0], -1, -1, -1)
                else:
                    raise ValueError("projection_mask batch size does not match image batch size.")

            mask_nchw = mask_tensor.permute(0, 3, 1, 2).contiguous()
            if mask_nchw.shape[2] != image_alpha.shape[1] or mask_nchw.shape[3] != image_alpha.shape[2]:
                mask_nchw = torch.nn.functional.interpolate(
                    mask_nchw,
                    size=(image_alpha.shape[1], image_alpha.shape[2]),
                    mode="bilinear",
                    align_corners=False,
                )
            image_alpha = image_alpha * mask_nchw.permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0)

        world_to_camera = torch.from_numpy(_pose_to_w2c(camera["pose_c2w"])).to(device=device, dtype=torch.float32)

        image_rgb_nchw = image_rgb.permute(0, 3, 1, 2).contiguous()
        image_alpha_nchw = image_alpha.permute(0, 3, 1, 2).contiguous()

        render_height = int(image_rgb_nchw.shape[2])
        render_width = int(image_rgb_nchw.shape[3])

        projection_mode = _camera_projection_mode(camera)
        aspect = float(render_width) / float(render_height)
        ortho_size = float(camera.get("orthographic_size", camera.get("ortho_size", 2.0)))
        if not np.isfinite(ortho_size) or ortho_size <= 0.0:
            ortho_size = 2.0

        world_position_flat = world_position.reshape(1, size * size, 3)
        ones = torch.ones((1, size * size, 1), device=device, dtype=torch.float32)
        world_position_h = torch.cat([world_position_flat, ones], dim=-1)
        camera_position_h = world_position_h @ world_to_camera.T

        x = camera_position_h[..., 0]
        y = camera_position_h[..., 1]
        z_camera = camera_position_h[..., 2]
        z = (-z_camera).clamp(min=1.0e-8)

        in_front = (z_camera < 0.0).float().view(1, size, size, 1)
        uv_mask = uv_mask * in_front.permute(0, 3, 1, 2)

        if projection_mode == "ortho":
            half_height = max(0.5 * ortho_size, 1.0e-8)
            half_width = max(half_height * aspect, 1.0e-8)
            x_ndc = x / half_width
            y_ndc = y / half_height
        else:
            fov_y = math.radians(float(camera.get("vertical_fov_deg", camera.get("fov_y_deg", 60.0))))
            tan_half = math.tan(0.5 * fov_y)
            x_ndc = (x / z) / (tan_half * aspect)
            y_ndc = (y / z) / tan_half

        sample_grid = torch.stack([x_ndc, -y_ndc], dim=-1).reshape(1, size, size, 2)
        sampled_rgb = torch.nn.functional.grid_sample(
            image_rgb_nchw,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled_alpha = torch.nn.functional.grid_sample(
            image_alpha_nchw,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).clamp(0.0, 1.0)

        u_pixels = (x_ndc + 1.0) * 0.5 * (render_width - 1)
        v_pixels = (1.0 - y_ndc) * 0.5 * (render_height - 1)

        projector_vertices_h = torch.cat(
            [vertices, torch.ones((1, vertices.shape[1], 1), device=device, dtype=torch.float32)],
            dim=-1,
        )
        projector_vertices_camera = projector_vertices_h @ world_to_camera.T
        projector_x = projector_vertices_camera[..., 0]
        projector_y = projector_vertices_camera[..., 1]
        projector_z_camera = projector_vertices_camera[..., 2]
        projector_depth = (-projector_z_camera).clamp(min=1.0e-8)

        if projection_mode == "ortho":
            half_height = max(0.5 * ortho_size, 1.0e-8)
            half_width = max(half_height * aspect, 1.0e-8)
            projector_x_ndc = projector_x / half_width
            projector_y_ndc = projector_y / half_height
        else:
            fov_y = math.radians(float(camera.get("vertical_fov_deg", camera.get("fov_y_deg", 60.0))))
            tan_half = math.tan(0.5 * fov_y)
            projector_x_ndc = (projector_x / projector_depth) / (tan_half * aspect)
            projector_y_ndc = (projector_y / projector_depth) / tan_half

        projector_x_pixels = (projector_x_ndc + 1.0) * 0.5 * (render_width - 1)
        projector_y_pixels = (1.0 - projector_y_ndc) * 0.5 * (render_height - 1)
        projector_raster_vertices = torch.stack(
            [projector_x_pixels, projector_y_pixels, projector_depth],
            dim=-1,
        )

        projector_triangle_index = drtk.rasterize(
            projector_raster_vertices,
            faces,
            height=render_height,
            width=render_width,
        )
        projector_render = drtk.render(projector_raster_vertices, faces, projector_triangle_index)
        projector_barycentric = (
            projector_render[1] if isinstance(projector_render, tuple) and len(projector_render) >= 2 else projector_render
        )
        if projector_barycentric is None:
            raise RuntimeError("DRTK camera render() did not return barycentrics.")

        projector_depth_image = drtk.interpolate(
            projector_depth.unsqueeze(-1),
            faces,
            projector_triangle_index,
            projector_barycentric,
        )
        if projector_depth_image.ndim == 4 and projector_depth_image.shape[-1] == 1:
            projector_depth_image = projector_depth_image.permute(0, 3, 1, 2)
        elif projector_depth_image.ndim != 4 or projector_depth_image.shape[1] != 1:
            raise RuntimeError(f"Unexpected projector depth image shape: {tuple(projector_depth_image.shape)}")

        projector_background = (projector_triangle_index == -1).unsqueeze(1)
        projector_depth_image = torch.where(
            projector_background,
            torch.full_like(projector_depth_image, float("inf")),
            projector_depth_image,
        )

        projector_grid = _pixel_grid_from_uv(u_pixels, v_pixels, render_width, render_height).reshape(1, size, size, 2)
        z_buffer = torch.nn.functional.grid_sample(
            projector_depth_image,
            projector_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        texture_depth = z.view(1, size, size, 1).permute(0, 3, 1, 2)
        uv_valid = (uv_triangle_index != -1).float().unsqueeze(1)
        valid_depth = torch.isfinite(z_buffer).float() * uv_valid
        visibility = (texture_depth <= (z_buffer + 1.0e-2)).float() * valid_depth

        occluded = (texture_depth > (z_buffer + 1.0e-2)).float() * valid_depth
        if self.DEFAULT_OCCLUSION_DILATION > 0:
            occluded = _dilate_mask(occluded, self.DEFAULT_OCCLUSION_DILATION) * uv_valid

        uv_boundary = (uv_valid - _erode_mask(uv_valid, 1)).clamp(0.0, 1.0)
        seam_guard_radius = self.DEFAULT_OCCLUSION_DILATION + self.DEFAULT_OCCLUSION_FADE
        seam_guard = _dilate_mask(uv_boundary, seam_guard_radius) if seam_guard_radius > 0 else uv_boundary
        unlock_radius = max(1, seam_guard_radius // 2) if seam_guard_radius > 0 else 1
        occlusion_influence = _dilate_mask(occluded, unlock_radius).clamp(0.0, 1.0)
        overlap_allowed = (uv_valid * ((1.0 - seam_guard) + seam_guard * occlusion_influence)).clamp(0.0, 1.0)

        if self.DEFAULT_OCCLUSION_FADE > 0:
            occluded_soft = (
                _blur_mask(occluded * valid_depth, self.DEFAULT_OCCLUSION_FADE)
                / _blur_mask(valid_depth, self.DEFAULT_OCCLUSION_FADE).clamp(min=1.0e-6)
            ).clamp(0.0, 1.0) * valid_depth
        else:
            occluded_soft = occluded.clamp(0.0, 1.0)

        visibility_soft = (visibility * (1.0 - occluded_soft)).clamp(0.0, 1.0)
        visibility = visibility * (1.0 - overlap_allowed) + visibility_soft * overlap_allowed
        uv_mask = uv_mask * _smoothstep(0.0, 1.0, visibility)
        uv_mask = _ensure_mask_shape(uv_mask)

        if self.DEFAULT_EDGE_FADE_PIXELS > 0:
            distance_to_edge = torch.minimum(
                torch.minimum(u_pixels, (render_width - 1) - u_pixels),
                torch.minimum(v_pixels, (render_height - 1) - v_pixels),
            )
            distance_to_edge = distance_to_edge.reshape(1, size, size, 1).clamp(min=0.0)
            edge_weight = _smoothstep(
                torch.zeros_like(distance_to_edge),
                torch.full_like(distance_to_edge, float(self.DEFAULT_EDGE_FADE_PIXELS)),
                distance_to_edge,
            ).pow(self.DEFAULT_EDGE_FADE_POWER)
        else:
            edge_weight = torch.ones((1, size, size, 1), device=device, dtype=torch.float32)

        soft_dot = 0.2
        dot_normalized = (normal_dot.clamp(min=-soft_dot, max=1.0) + soft_dot) / (1.0 + soft_dot)
        dot_normalized = _smoothstep(torch.zeros_like(dot_normalized), torch.ones_like(dot_normalized), dot_normalized)
        dot_weight = (dot_normalized - normal_fade_min) / (1.0 - normal_fade_min + 1.0e-2)
        dot_weight = _smoothstep(torch.zeros_like(dot_weight), torch.ones_like(dot_weight), dot_weight)
        normal_weight = dot_weight.clamp(0.0, 1.0).pow(normal_fade_power)

        alpha = (uv_mask.permute(0, 2, 3, 1) * edge_weight * normal_weight).permute(0, 3, 1, 2)
        alpha = (alpha * sampled_alpha * float(opacity)).clamp(0.0, 1.0)
        projected_rgb = sampled_rgb[:, 0:3].clamp(0.0, 1.0)

        if int(dilation_pixels) > 0:
            projected_rgb, alpha = _dilate_color_by_alpha(projected_rgb, alpha, int(dilation_pixels))

        projected_rgb = projected_rgb * alpha

        base_image = _get_basecolor_image_from_trimesh(trimesh_for_projection, size)
        base_image_np = np.asarray(base_image).astype(np.float32) / 255.0
        base_texture = torch.from_numpy(base_image_np).to(device=device).permute(2, 0, 1).unsqueeze(0)

        if bool(apply_to_trimesh):
            output_rgb = base_texture[:, 0:3] * (1.0 - alpha) + projected_rgb
            output_alpha = base_texture[:, 3:4] + alpha * (1.0 - base_texture[:, 3:4])
            output_texture = torch.cat([output_rgb, output_alpha], dim=1).clamp(0.0, 1.0)

            output_image_np = (
                output_texture[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0 + 0.5
            ).astype(np.uint8)
            output_image = Image.fromarray(output_image_np, mode="RGBA")
            updated_trimesh = _set_basecolor_image_on_trimesh(trimesh_for_projection.copy(), output_image)
            texture_map = _to_comfy_image_tensor(output_image)
        else:
            updated_trimesh = trimesh_for_projection
            texture_map = _to_comfy_image_tensor(base_image)

        projected_layer = projected_rgb.permute(0, 2, 3, 1).contiguous().detach().cpu()
        projected_alpha = alpha[0, 0].detach().cpu()
        return (updated_trimesh, texture_map, projected_layer, projected_alpha)

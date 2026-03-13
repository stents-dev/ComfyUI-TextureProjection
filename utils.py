import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import trimesh as Trimesh


def _euler_xyz_deg_to_rotmat(rotation_x_deg: float, rotation_y_deg: float, rotation_z_deg: float) -> np.ndarray:
    rx = math.radians(float(rotation_x_deg))
    ry = math.radians(float(rotation_y_deg))
    rz = math.radians(float(rotation_z_deg))

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
        dtype=np.float32,
    )
    rot_y = np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=np.float32,
    )
    rot_z = np.array(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return (rot_z @ rot_y @ rot_x).astype(np.float32)


def _look_at_pose(camera_position: np.ndarray, target_point: np.ndarray, up: np.ndarray = None) -> np.ndarray:
    if up is None:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    camera_position = np.asarray(camera_position, dtype=np.float32)
    target_point = np.asarray(target_point, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    forward = target_point - camera_position
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1.0e-6:
        forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        forward = forward / forward_norm

    right = np.cross(forward, up)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1.0e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        right = right / right_norm

    true_up = np.cross(right, forward)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.stack([right, true_up, -forward], axis=1).astype(np.float32)
    pose[:3, 3] = camera_position
    return pose


def _pose_to_w2c(pose_c2w: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(pose_c2w, dtype=np.float32)).astype(np.float32)


def _compute_intrinsics_from_fovy(vertical_fov_deg: float, width: int, height: int):
    fov_y = math.radians(float(vertical_fov_deg))
    fy = 0.5 * float(height) / math.tan(0.5 * fov_y)
    fx = fy * (float(width) / float(height))
    cx = 0.5 * (float(width) - 1.0)
    cy = 0.5 * (float(height) - 1.0)
    return float(fx), float(fy), float(cx), float(cy)


def _compute_intrinsics_from_ortho_size(orthographic_size: float, width: int, height: int):
    half_height = max(0.5 * float(orthographic_size), 1.0e-8)
    half_width = max(half_height * (float(width) / float(height)), 1.0e-8)
    fx = 0.5 * (float(width) - 1.0) / half_width
    fy = 0.5 * (float(height) - 1.0) / half_height
    cx = 0.5 * (float(width) - 1.0)
    cy = 0.5 * (float(height) - 1.0)
    return float(fx), float(fy), float(cx), float(cy)


def _camera_projection_mode(camera: dict, default: str = "perspective") -> str:
    mode = str((camera or {}).get("projection", (camera or {}).get("projection_type", default))).strip().lower()
    if mode in ("ortho", "orthographic"):
        return "ortho"
    return "perspective"


def _as_trimesh(mesh):
    if isinstance(mesh, Trimesh.Trimesh):
        return mesh
    if isinstance(mesh, Trimesh.Scene):
        geometries = [geometry for geometry in mesh.geometry.values() if isinstance(geometry, Trimesh.Trimesh)]
        if not geometries:
            raise ValueError("Scene contains no Trimesh geometry.")
        return Trimesh.util.concatenate(geometries)
    raise TypeError(f"Unsupported mesh type: {type(mesh)}")


def smooth_normals_across_duplicate_positions(
    mesh: Trimesh.Trimesh,
    pos_tol: float = 1e-6,
    area_weighted: bool = True,
) -> np.ndarray:
    """
    Compute smooth vertex normals, then re-average normals for vertices that
    share the same position (within pos_tol).
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertex_count = int(vertices.shape[0])
    if vertex_count == 0:
        return np.zeros((0, 3), dtype=np.float32)

    faces = np.asarray(mesh.faces, dtype=np.int64)
    valid_face_mask = (
        (faces.ndim == 2)
        and (faces.shape[1] == 3)
        and (faces.size > 0)
    )
    if valid_face_mask:
        valid_face_mask = (
            (faces[:, 0] >= 0)
            & (faces[:, 1] >= 0)
            & (faces[:, 2] >= 0)
            & (faces[:, 0] < vertex_count)
            & (faces[:, 1] < vertex_count)
            & (faces[:, 2] < vertex_count)
        )
        faces = faces[valid_face_mask]
    else:
        faces = np.zeros((0, 3), dtype=np.int64)

    normals = None
    if not area_weighted:
        try:
            normals = Trimesh.smoothing.get_vertices_normals(mesh).astype(np.float64, copy=False)
        except Exception:
            normals = None

    if normals is None:
        accumulated = np.zeros((vertex_count, 3), dtype=np.float64)
        if faces.shape[0] > 0:
            face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
            if face_normals.shape[0] != np.asarray(mesh.faces).shape[0]:
                face_normals = np.zeros((faces.shape[0], 3), dtype=np.float64)
            else:
                face_normals = face_normals[valid_face_mask]
            face_normals = np.nan_to_num(face_normals, nan=0.0, posinf=0.0, neginf=0.0)

            if area_weighted:
                face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
                if face_areas.shape[0] != np.asarray(mesh.faces).shape[0]:
                    face_areas = np.ones((faces.shape[0],), dtype=np.float64)
                else:
                    face_areas = face_areas[valid_face_mask]
                face_areas = np.nan_to_num(face_areas, nan=0.0, posinf=0.0, neginf=0.0)
                weighted_face_normals = face_normals * face_areas[:, None]
            else:
                weighted_face_normals = face_normals

            np.add.at(accumulated, faces[:, 0], weighted_face_normals)
            np.add.at(accumulated, faces[:, 1], weighted_face_normals)
            np.add.at(accumulated, faces[:, 2], weighted_face_normals)

        accumulated_norm = np.linalg.norm(accumulated, axis=1, keepdims=True)
        normals = accumulated / (accumulated_norm + 1e-12)

        bad = accumulated_norm[:, 0] <= 1e-20
        if np.any(bad):
            normals[bad] = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    tolerance = max(float(pos_tol), 1e-12)
    quantized_positions = np.round(vertices / tolerance).astype(np.int64)
    keys = np.core.records.fromarrays(quantized_positions.T, names="x,y,z", formats="i8,i8,i8")

    order = np.argsort(keys, kind="mergesort")
    keys_sorted = keys[order]
    normals_sorted = normals[order]

    same_group = (
        (keys_sorted["x"][1:] == keys_sorted["x"][:-1])
        & (keys_sorted["y"][1:] == keys_sorted["y"][:-1])
        & (keys_sorted["z"][1:] == keys_sorted["z"][:-1])
    )
    starts = np.concatenate(([0], np.nonzero(~same_group)[0] + 1))
    ends = np.concatenate((starts[1:], [len(order)]))

    output_normals = normals.copy()
    for start, end in zip(starts, ends):
        indices = order[start:end]
        average = normals_sorted[start:end].sum(axis=0)
        average /= np.linalg.norm(average) + 1e-12
        output_normals[indices] = average

    return output_normals.astype(np.float32)


def _rgb_uint8_to_comfy(rgb_uint8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(rgb_uint8.astype(np.float32) / 255.0).unsqueeze(0)


def _to_comfy_image_tensor(image: Image.Image) -> torch.Tensor:
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA")
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    return torch.from_numpy(array).unsqueeze(0)


def _smoothstep(edge0, edge1, x):
    t = ((x - edge0) / (edge1 - edge0 + 1.0e-8)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _ensure_mask_shape(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3:
        return mask.unsqueeze(1)
    return mask


def _blur_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = radius * 2 + 1
    mask = F.avg_pool2d(mask, kernel_size=(1, kernel), stride=1, padding=(0, radius))
    mask = F.avg_pool2d(mask, kernel_size=(kernel, 1), stride=1, padding=(radius, 0))
    return mask


def _erode_mask(mask: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    result = mask
    for _ in range(int(iterations)):
        result = 1.0 - F.max_pool2d(1.0 - result, kernel_size=3, stride=1, padding=1)
    return result


def _dilate_mask(mask: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    result = mask
    for _ in range(int(iterations)):
        result = F.max_pool2d(result, kernel_size=3, stride=1, padding=1)
    return result


def _pixel_grid_from_uv(u_pixels: torch.Tensor, v_pixels: torch.Tensor, width: int, height: int) -> torch.Tensor:
    grid_x = (u_pixels / max(1.0, float(width - 1))) * 2.0 - 1.0
    grid_y = (v_pixels / max(1.0, float(height - 1))) * 2.0 - 1.0
    return torch.stack([grid_x, grid_y], dim=-1)


def _dilate_color_by_alpha(rgb: torch.Tensor, alpha: torch.Tensor, iterations: int):
    result_rgb = rgb
    result_alpha = alpha
    for _ in range(int(iterations)):
        alpha_pool, indices = F.max_pool2d(result_alpha, kernel_size=3, stride=1, padding=1, return_indices=True)
        batch, channels, height, width = result_rgb.shape
        rgb_flat = result_rgb.view(batch, channels, -1)
        gather_idx = indices.view(batch, 1, -1).expand(batch, channels, -1)
        rgb_fill = torch.gather(rgb_flat, 2, gather_idx).view(batch, channels, height, width)

        replace_mask = (result_alpha < 1.0e-6).float()
        result_rgb = result_rgb * (1.0 - replace_mask) + rgb_fill * replace_mask
        result_alpha = torch.maximum(result_alpha, alpha_pool)

    return result_rgb, result_alpha


def _get_basecolor_image_from_trimesh(mesh: Trimesh.Trimesh, size: int) -> Image.Image:
    base_image = None
    material = getattr(mesh.visual, "material", None) if getattr(mesh, "visual", None) is not None else None
    if material is not None:
        for attr in ("image", "baseColorTexture"):
            candidate = getattr(material, attr, None)
            if candidate is not None:
                base_image = candidate
                break

    if base_image is None:
        return Image.new("RGBA", (int(size), int(size)), (255, 255, 255, 255))

    if isinstance(base_image, Image.Image):
        return base_image.convert("RGBA").resize((int(size), int(size)), Image.BILINEAR)

    try:
        return Image.fromarray(np.asarray(base_image)).convert("RGBA").resize(
            (int(size), int(size)),
            Image.BILINEAR,
        )
    except Exception:
        return Image.new("RGBA", (int(size), int(size)), (255, 255, 255, 255))


def _set_basecolor_image_on_trimesh(mesh: Trimesh.Trimesh, image: Image.Image) -> Trimesh.Trimesh:
    updated_mesh = mesh
    uv = np.asarray(updated_mesh.visual.uv, dtype=np.float32)

    try:
        material = getattr(updated_mesh.visual, "material", None)
        if material is not None:
            if hasattr(material, "image"):
                material.image = image
            elif hasattr(material, "baseColorTexture"):
                material.baseColorTexture = image
            else:
                updated_mesh.visual = Trimesh.visual.texture.TextureVisuals(uv=uv, image=image, material=material)
        else:
            updated_mesh.visual = Trimesh.visual.texture.TextureVisuals(uv=uv, image=image)
    except Exception:
        updated_mesh.visual = Trimesh.visual.texture.TextureVisuals(uv=uv, image=image)

    return updated_mesh

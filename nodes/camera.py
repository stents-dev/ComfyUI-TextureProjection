import numpy as np

from ..utils import (
    _compute_intrinsics_from_fovy,
    _compute_intrinsics_from_ortho_size,
    _euler_xyz_deg_to_rotmat,
    _look_at_pose,
)


def _normalize_projection_type(projection_type):
    return "ortho" if str(projection_type).strip().lower() == "orthographic" else "perspective"


def _sanitize_camera_scalar_inputs(vertical_fov_deg, orthographic_size, image_width, image_height):
    width = int(image_width)
    height = int(image_height)

    ortho_size = float(orthographic_size)
    if not np.isfinite(ortho_size) or ortho_size <= 0.0:
        ortho_size = 2.0

    fov_y_deg = float(vertical_fov_deg)
    if not np.isfinite(fov_y_deg):
        fov_y_deg = 45.0

    return width, height, fov_y_deg, ortho_size


def _build_camera_parameters(
    pose_c2w,
    camera_position,
    target_point,
    projection_type,
    vertical_fov_deg,
    orthographic_size,
    image_width,
    image_height,
):
    width, height, fov_y_deg, ortho_size = _sanitize_camera_scalar_inputs(
        vertical_fov_deg,
        orthographic_size,
        image_width,
        image_height,
    )
    projection_mode = _normalize_projection_type(projection_type)

    if projection_mode == "ortho":
        fx, fy, cx, cy = _compute_intrinsics_from_ortho_size(ortho_size, width, height)
    else:
        fx, fy, cx, cy = _compute_intrinsics_from_fovy(fov_y_deg, width, height)

    return {
        "pose_c2w": np.asarray(pose_c2w, dtype=np.float32),
        "camera_position": np.asarray(camera_position, dtype=np.float32),
        "target_point": np.asarray(target_point, dtype=np.float32),
        "focal_point": np.asarray(target_point, dtype=np.float32),
        "projection": projection_mode,
        "projection_type": projection_mode,
        "fov_y_deg": fov_y_deg,
        "vertical_fov_deg": fov_y_deg,
        "ortho_size": ortho_size,
        "orthographic_size": ortho_size,
        "render_width": width,
        "render_height": height,
        "image_width": width,
        "image_height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }


def _build_look_at_camera(
    camera_position,
    target_point,
    projection_type,
    vertical_fov_deg,
    orthographic_size,
    image_width,
    image_height,
):
    pose_c2w = _look_at_pose(camera_position, target_point)
    return _build_camera_parameters(
        pose_c2w=pose_c2w,
        camera_position=camera_position,
        target_point=target_point,
        projection_type=projection_type,
        vertical_fov_deg=vertical_fov_deg,
        orthographic_size=orthographic_size,
        image_width=image_width,
        image_height=image_height,
    )


class CreateCameraParameters:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_mode": (["look_at", "euler"], {"default": "look_at"}),
                "position_x": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "position_y": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "position_z": ("FLOAT", {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "target_x": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "target_y": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "target_z": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "rotation_x_deg": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "rotation_y_deg": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "rotation_z_deg": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "projection_type": (["perspective", "orthographic"], {"default": "perspective"}),
                "vertical_fov_deg": ("FLOAT", {"default": 40.0, "min": 5.0, "max": 120.0, "step": 0.1}),
                "orthographic_size": ("FLOAT", {"default": 2.0, "min": 0.0001, "max": 10000.0, "step": 0.01}),
                "image_width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "image_height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
            }
        }

    RETURN_TYPES = ("CAM_PARAMS",)
    RETURN_NAMES = ("camera",)
    FUNCTION = "create"
    CATEGORY = "3D/Texture Projection"

    def create(
        self,
        pose_mode,
        position_x,
        position_y,
        position_z,
        target_x,
        target_y,
        target_z,
        rotation_x_deg,
        rotation_y_deg,
        rotation_z_deg,
        projection_type,
        vertical_fov_deg,
        orthographic_size,
        image_width,
        image_height,
    ):
        camera_position = np.array([position_x, position_y, position_z], dtype=np.float32)
        target_point = np.array([target_x, target_y, target_z], dtype=np.float32)

        if pose_mode == "look_at":
            pose_c2w = _look_at_pose(camera_position, target_point)
        else:
            rotation = _euler_xyz_deg_to_rotmat(rotation_x_deg, rotation_y_deg, rotation_z_deg)
            pose_c2w = np.eye(4, dtype=np.float32)
            pose_c2w[:3, :3] = rotation
            pose_c2w[:3, 3] = camera_position

        camera = _build_camera_parameters(
            pose_c2w=pose_c2w,
            camera_position=camera_position,
            target_point=target_point,
            projection_type=projection_type,
            vertical_fov_deg=vertical_fov_deg,
            orthographic_size=orthographic_size,
            image_width=image_width,
            image_height=image_height,
        )
        return (camera,)


class MultiViewCamera:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "offset": ("FLOAT", {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "vertical_offset": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "projection_type": (["perspective", "orthographic"], {"default": "perspective"}),
                "fov": ("FLOAT", {"default": 40.0, "min": 5.0, "max": 120.0, "step": 0.1}),
                "orthographic_size": ("FLOAT", {"default": 2.0, "min": 0.0001, "max": 10000.0, "step": 0.01}),
                "image_width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "image_height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
            }
        }

    RETURN_TYPES = ("CAM_PARAMS", "CAM_PARAMS", "CAM_PARAMS", "CAM_PARAMS", "CAM_PARAMS", "CAM_PARAMS")
    RETURN_NAMES = ("front", "back", "left", "right", "bottom", "top")
    FUNCTION = "create"
    CATEGORY = "3D/Texture Projection"
    DESCRIPTION = "Creates six look-at camera parameter sets for front, back, left, right, bottom, and top views."

    def create(
        self,
        offset,
        vertical_offset,
        projection_type,
        fov,
        orthographic_size,
        image_width,
        image_height,
    ):
        target_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        offset_value = float(offset)
        vertical_offset_value = float(vertical_offset)

        camera_positions = {
            "front": np.array([0.0, vertical_offset_value, offset_value], dtype=np.float32),
            "back": np.array([0.0, vertical_offset_value, -offset_value], dtype=np.float32),
            "left": np.array([-offset_value, vertical_offset_value, 0.0], dtype=np.float32),
            "right": np.array([offset_value, vertical_offset_value, 0.0], dtype=np.float32),
            "bottom": np.array([0.0, -offset_value, 0.0], dtype=np.float32),
            "top": np.array([0.0, offset_value, 0.0], dtype=np.float32),
        }

        outputs = []
        for view_name in ("front", "back", "left", "right", "bottom", "top"):
            outputs.append(
                _build_look_at_camera(
                    camera_position=camera_positions[view_name],
                    target_point=target_point,
                    projection_type=projection_type,
                    vertical_fov_deg=fov,
                    orthographic_size=orthographic_size,
                    image_width=image_width,
                    image_height=image_height,
                )
            )

        return tuple(outputs)

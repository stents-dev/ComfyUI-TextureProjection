# ComfyUI-TextureProjection

A minimal ComfyUI custom node pack for camera-based texture projection onto UV-mapped trimeshes.

These nodes make use of DRTK for rendering (https://github.com/facebookresearch/DRTK/)

1. Render the model
2. Enhance the render using SDXL, Flux, Qwen-Image-Edit or any model of your choice
3. Project the texture back onto the model

![example image](example_workflows/images/example.jpg)

The UV projection step includes several quality controls to help multiple views combine cleanly:

- Overlap handling with soft visibility fades, so competing views blend out more gracefully instead of leaving hard seams
- Edge fade
- Normal-based weighting, so surfaces facing the camera contribute more strongly while glancing angles fade out
- Optional projection masks and opacity control, for limiting where a view is allowed to write
- Texture dilation after projection, which helps fill tiny gaps and reduce fringe artifacts around projected regions

Please refer to the example_workflows folder for examples of using both SDXL + Controlnets or Flux.2 Klein.

## Nodes

- `TextureProjection Load GLB`
  - Loads a GLB file as a `TRIMESH`
- `TextureProjection Camera`
  - Creates a `CAM_PARAMS` bundle for use in rendering and projection nodes.
- `TextureProjection MultiView Camera`
  - Automatically constructs 6 camera vies for Front, Back, Left, Right, Top and Bottom.
- `TextureProjection Render Mesh`
  - Renders base color, depth and normals.
- `TextureProjection Project To UV`
  - Projects an input image from camera space back into UV space.
- `TextureProjection Trimesh To GLB`
  - Converts a `TRIMESH` into an in-memory `FILE_3D_GLB` object so it can be passed directly into native ComfyUI nodes such as `Preview3D` and `SaveGLB`.

## Requirements

This node pack expects a working ComfyUI environment with:

- PyTorch with CUDA support
- `numpy`
- `Pillow`
- `trimesh`
- `drtk`

## Installation

Clone or copy this folder into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone <your-repo-url> ComfyUI-TextureProjection
```

Install dependencies in the same Python environment that ComfyUI uses:

```bash
pip install -r requirements.txt
```

Install drtk: https://github.com/facebookresearch/DRTK

A small number of pre built wheels for windows can be found for `drtk` in the wheels folder, otherwise you will need to build it yourself (see https://drtk.xyz/installation/index.html).

Then restart ComfyUI.

## Notes

- The model must already have UVs for projection.
- CUDA is required for rendering and projection.

## License

MIT. See [LICENSE](LICENSE).

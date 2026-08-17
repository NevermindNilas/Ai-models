# Limbo-V1

**Anime monocular relative depth. 34.3 M parameters. Ready-to-run ONNX, opset 17.**

Original Arch: https://github.com/ByteDance-Seed/Depth-Anything-3
Original Arch License: Apache License 2.0

A fine-tune of [`depth-anything/DA3-SMALL`](https://huggingface.co/depth-anything/DA3-SMALL)
(Apache-2.0), trained by knowledge distillation on an anime corpus with a natural-image anchor
set constraining how much photographic ability it gives up. The published weights are a
0.75 / 0.25 weighted average ("model soup") of two fine-tuning runs.

The ONNX files need nothing but `onnxruntime` — no PyTorch, no `depth_anything_3`.

## Files

| file | dtype | batch | resolution | size | use it for |
| :-- | :-- | --: | :-- | --: | :-- |
| `Limbo_504x280_fp16.onnx` | fp16 | 1 | 504x280 (16:9) | 59.8 MiB | **default** for widescreen frames |
| `Limbo_504x280_fp32.onnx` | fp32 | 1 | 504x280 (16:9) | 95.8 MiB | reference precision, or runtimes without fp16 kernels |
| `Limbo_504x280_b4_fp16.onnx` | fp16 | 4 | 504x280 (16:9) | 49.3 MiB | video / batches — ~1.4x the throughput per frame |
| `Limbo_504x378_fp16.onnx` | fp16 | 1 | 504x378 (4:3) | 64.3 MiB | 4:3 sources (older TV anime) |
| `Limbo_504x378_fp32.onnx` | fp32 | 1 | 504x378 (4:3) | 95.8 MiB | 4:3, reference precision |
| `Limbo.safetensors` | fp16 | — | — | 65.5 MiB | the checkpoint, for re-export or further training |

SHA-256:

```
905ec184d401e4cc534a6e09a08f6d8a8587a417d091227053ce052d2ec89c8e  Limbo_504x280_fp32.onnx
31a0f8b38b531e19fca616336c638529eb0f3b41cde75dea8b6aad4d164df7d7  Limbo_504x280_fp16.onnx
49d0eb03bcc565f377c5ea3cc52d6e37de21d93b0d4d9954b8b0593380b0aad9  Limbo_504x280_b4_fp16.onnx
8cee16b11cafbe2d5f71f2cba95873dc774c9e78ac2a35952e9860ab2744657b  Limbo_504x378_fp32.onnx
b7453d2aad2543020553340aee0a9a99d1c00f155766b5ef223e43b3ab5ca155  Limbo_504x378_fp16.onnx
be4a8f22a134c683cef71913fdf981293f2a5e0b6510463c3442da1d1e825b46  Limbo.safetensors
```

Every file carries its own license, upstream notice and provenance in its metadata header — the
ONNX graphs in `metadata_props`, the checkpoint in the safetensors header — so the terms survive
being renamed or re-hosted:

```python
import onnx
print({p.key: p.value for p in onnx.load("Limbo_504x280_fp16.onnx", load_external_data=False).metadata_props})

from safetensors import safe_open
with safe_open("Limbo.safetensors", "pt") as f: print(f.metadata())
```

### The checkpoint

`Limbo.safetensors` is fp16, 443 tensors, with `soup_members` / `soup_weights` provenance in its
header. It is a plain state dict: loading it into a model needs the
[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) code and the `DA3-SMALL`
`config.json`, with `strict=False` (`head.scratch.output_conv2_aux.*` differs from upstream by
design). `load_state_dict` casts it to your model's dtype.

It is stored at fp16 rather than fp32 to stay one file: GitHub refuses any blob over 100 MiB and
the fp32 original is 130.8 MiB. That costs nothing measurable — re-scored on the frozen anime set,
the fp16 checkpoint gives `disp_mae` **0.48926** against the fp32 original's **0.48930**, with
every other metric identical to four decimals. If you need bit-exact fp32 for resumed training,
ask and it can be published separately.

### Download

```bash
curl -LO https://raw.githubusercontent.com/NevermindNilas/Ai-models/main/Limbo-V1/Limbo_504x280_fp16.onnx
```

## Quick start

```bash
pip install onnxruntime-directml numpy pillow   # Windows GPU (DirectML)
# pip install onnxruntime            # CPU only, any OS
# pip install onnxruntime-gpu        # CUDA / TensorRT
```

```python
import numpy as np, onnxruntime as ort
from PIL import Image

MEAN, STD = np.float32([0.485, 0.456, 0.406]), np.float32([0.229, 0.224, 0.225])

sess = ort.InferenceSession(
    "Limbo_504x280_fp16.onnx",
    providers=["DmlExecutionProvider", "CPUExecutionProvider"],  # or CUDA/TensorRT
)
print("EP:", sess.get_providers())     # only CPUExecutionProvider? you are on the slow path

inp = sess.get_inputs()[0]
_, _, H, W = inp.shape                 # 280, 504 — baked into the file
dt = np.float16 if "float16" in inp.type else np.float32

src = Image.open("frame.png").convert("RGB")
img = src.resize((W, H), Image.LANCZOS)          # LANCZOS: same filter the corpus was built with
x = ((np.asarray(img, np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None].astype(dt)

depth, conf = sess.run(["depth", "depth_conf"], {inp.name: x})   # (1,H,W) each

# depth is POSITIVE DEPTH. For the usual near-is-bright picture, invert it:
disp = 1.0 / np.clip(depth[0].astype(np.float32), 1e-6, None)
vis = (disp - disp.min()) / (disp.max() - disp.min())
Image.fromarray((vis * 255).astype(np.uint8)).resize(src.size, Image.BILINEAR).save("depth.png")
```

**Pick the file by aspect ratio.** 504x280 is 1.80, 504x378 is 1.333. A plain `.resize()` to a
mismatched shape stretches the image, which costs accuracy; centre-crop (or letterbox) to the
target ratio first:

```python
ar = W / H
sw, sh = src.size
if sw / sh > ar:                                  # too wide -> crop left/right
    new_w = int(round(sh * ar));  off = (sw - new_w) // 2
    src = src.crop((off, 0, off + new_w, sh))
else:                                             # too tall -> crop top/bottom
    new_h = int(round(sw / ar));  off = (sh - new_h) // 2
    src = src.crop((0, off, sw, off + new_h))
```

Letterboxed sources are the exception: crop the black bars off first, or the model spends its
depth range on them.

## I/O contract

| | name | shape | dtype | notes |
| :-- | :-- | :-- | :-- | :-- |
| input | `image` | `(B, 3, H, W)` | fp16 or fp32, matching the file | **ImageNet-normalised** RGB |
| output 0 | `depth` | `(B, H, W)` | same as input | **positive depth**, `exp` activation |
| output 1 | `depth_conf` | `(B, H, W)` | same as input | confidence, `exp(x) + 1` |

Two mistakes to avoid, both easy to make:

1. The input is **ImageNet-normalised**, not `[0,1]` and not BGR: `(rgb/255 - mean) / std`.
2. The output is **depth**, not disparity. Higher = further away. Invert it before applying the
   usual colour maps, or your depth map reads inside-out.

Relative depth only: the values carry no metric scale, and are comparable within one image, not
across images. Each batch slot is an independent image — the model's multi-view axis is fixed to
one view inside the graph, so batching cannot mix frames.

## Quality

Scored on a frozen anime evaluation set (390 held-out anime frames, 389 after label QC) and, for
forgetting, on the first 400 images of the NYUv2 test split with real Kinect ground truth.
`disp_mae` is the primary metric: mean absolute error of robust-normalised disparity after
per-image affine alignment, lower is better.

| | anime `disp_mae` ↓ | anime AbsRel ↓ | anime δ1 ↑ | NYUv2 AbsRel ↓ | NYUv2 δ1 ↑ |
| :-- | --: | --: | --: | --: | --: |
| fp32 training checkpoint (reference) | 0.48930 | 0.27123 | 0.66462 | 0.13237 | 0.82909 |
| `Limbo.safetensors` (fp16) | 0.48926 | 0.27122 | 0.66467 | — | — |
| `Limbo_504x280_fp32.onnx` | 0.48992 | 0.27168 | 0.66434 | — | — |
| `Limbo_504x280_fp16.onnx` | 0.48987 | 0.27187 | 0.66424 | — | — |
| `Limbo_504x280_b4_fp16.onnx` | 0.48986 | — | — | — | — |
| `Limbo_504x378_fp32.onnx` | — | — | — | 0.13239 | 0.82916 |
| `Limbo_504x378_fp16.onnx` | — | — | — | 0.13232 | 0.82933 |

Every export lands within **0.13 %** of the reference on the primary metric, and within 0.4 % on
the worst of the fourteen tracked metrics. fp16 costs nothing measurable.

**Read the anime numbers honestly:** they measure *agreement with the teacher that produced the
training targets*, because anime has no ground-truth depth. A student that faithfully copies the
teacher's mistakes scores well on them. The NYUv2 column is real ground truth and shows how much
photographic ability was traded away — very little.

## Performance

RTX 3090, onnxruntime 1.27 (DirectML build), 504x280, one session per process, median of 80 runs:

| | DirectML | CPU EP |
| :-- | --: | --: |
| fp32, batch 1 | 9.1 ms | 82 ms |
| fp16, batch 1 | **6.4 ms** | 92 ms |
| fp16, batch 4 (this file, IO-bound) | **4.6 ms/frame** (219 fps) | — |

For reference, the same model in eager PyTorch fp32 on the same GPU is 14.5 ms. Opset 17 is
deliberate: every newer opset introduces an operator DirectML has no kernel for (`ScatterND`-18,
`Gelu`-20, `Attention`-23), which splits the graph and costs 3–16x. At opset 17 the whole network
runs as a single fused DirectML node.

## Constraints

* **Resolution is baked in.** One file per shape. The backbone derives its positional-embedding
  interpolation, RoPE table and patch grid from Python ints read off the input shape, and tracing
  freezes them — a dynamic-axis export errors out at the positional-embedding add rather than
  returning anything usable at another size. Resize to the file's exact `H x W`.
* Both sides must be multiples of 14 (the patch size). 504 is the longest-side resolution the
  upstream pipeline defaults to; the backbone's pretraining grid is 518x518.
* **Batch is static too**, for the same reason. Use the `b4` file for batches of 4.
* The graph is the monocular depth path only: the camera-pose branch and the auxiliary "ray" head
  are removed. `depth` is bit-identical to the full model's depth output.
* Trained on stylised animation. Not a metric depth model, not a photographic depth model, and
  not fit for anything safety-critical.

## Provenance

| | |
| :-- | :-- |
| base weights | [`depth-anything/DA3-SMALL`](https://huggingface.co/depth-anything/DA3-SMALL) — Apache-2.0 — fine-tuned; its values are present in these files |
| model code | [`ByteDance-Seed/Depth-Anything-3`](https://github.com/ByteDance-Seed/Depth-Anything-3) — Apache-2.0 — topology and exported op sequence |
| training data | an anime corpus of publicly available broadcast frames, illustrations and background art (~24.6 k images), plus a photographic anchor set used only to limit forgetting |
| evaluation | frozen anime split with pseudo-label targets; NYUv2 test split (real ground truth); CC0 anime alpha mattes for a teacher-free boundary metric |

No non-commercially-licensed model was involved in producing these weights.

## License

**Dual.** [`LICENSE.MD`](LICENSE.MD) in this directory is the authoritative grant. In short:

* by default, **CC BY-NC 4.0** — the unmodified legal code is at the repository root,
  [`LICENSE.md`](../LICENSE.md);
* **anyone using these weights through
  [TheAnimeScripter](https://github.com/NevermindNilas/TheAnimeScripter)** — users included — has
  a full commercial and closed-source grant, with no further permission needed;
* the upstream Depth Anything 3 material remains under **Apache-2.0** — verbatim copy in
  [`LICENSE-APACHE-2.0.MD`](LICENSE-APACHE-2.0.MD), retained copyright notices and the statement
  of modifications in section 4 of `LICENSE.MD`. Those notices travel with the files if you
  redistribute them; simply running the model does not trigger anything.

Not affiliated with or endorsed by ByteDance, the Depth Anything authors, or Meta.

## Citation

If you use these weights, please also cite the upstream work they derive from:

```bibtex
@article{depthanything3,
  title={Depth Anything 3: Recovering the visual space from any views},
  author={Haotong Lin and Sili Chen and Jun Hao Liew and Donny Y. Chen and Zhenyu Li and Guang Shi and Jiashi Feng and Bingyi Kang},
  journal={arXiv preprint arXiv:2511.10647},
  year={2025}
}
```

# Cyte-V1-SuperUltraCompact

**Ultra-compact 2x anime upscaler. 45,156 parameters.**

Original Arch: https://github.com/xinntao/Real-ESRGAN (SRVGGNetCompact, "SuperUltraCompact")
Original Arch License: BSD-3-Clause, copyright (c) 2021 Xintao Wang -
see [LICENSE-BSD-3-CLAUSE.MD](LICENSE-BSD-3-CLAUSE.MD).

The trained model weights in this directory are an original work and are licensed under
CC BY-NC 4.0, with an additional commercial-use grant for TheAnimeScripter users -
see [LICENSE.MD](LICENSE.MD).

## Training

| Detail | Value |
| :-- | :-- |
| Architecture | SRVGGNetCompact ("SuperUltraCompact"), 24nf / 8nc, 2x, RGB |
| Parameters | 45,156 |
| Datasets | [Zarxrax/ModernAnimation1080_v3](https://huggingface.co/datasets/Zarxrax/ModernAnimation1080_v3), [umzi/DigitalArtV3](https://huggingface.co/datasets/umzi/DigitalArtV3), [umzi/DigitalArt_v5](https://huggingface.co/datasets/umzi/DigitalArt_v5) |
| Iterations | Undisclosed |
| OTF (on-the-fly degradations) | Yes |

## Files

| file | dtype | opset | size | use it for |
| :-- | :-- | --: | --: | :-- |
| `2x-Cyte-V1-SuperUltraCompact.pth` | fp32 | - | 0.36 MiB | the checkpoint, for re-export or further training |
| `Cyte-V1_2x_fp32_op17_slim.onnx` | fp32 | 17 | 0.18 MiB | reference precision, or runtimes without fp16 kernels |
| `Cyte-V1_2x_fp16_op17_slim.onnx` | fp16 | 17 | 0.09 MiB | **default** for most GPUs |

Dynamic spatial axes (`1x3xHxW` in, `1x3xH2xW2` out). Verified with a `(1, 3, 128, 128)`
input on both files.

SHA-256:

```
d4eea1e072f74f42e864d8a76e10a3df48b5f996f4f9db0a613799455a1d86f8  2x-Cyte-V1-SuperUltraCompact.pth
4dbe59c45a8c3a5f7dbe28fd271a937bbb354a4cc8800c5024ef51c363a21b95  Cyte-V1_2x_fp32_op17_slim.onnx
0b58fa4df2387c25557841e97317796ad51cbbc0756bc203b23efd4acabf42d5  Cyte-V1_2x_fp16_op17_slim.onnx
```

## Quick start

The `.onnx` files need nothing but `onnxruntime` - no PyTorch, no export step:

```bash
pip install onnxruntime-directml numpy pillow  # Windows GPU (DirectML)
# pip install onnxruntime-gpu                   # CUDA / TensorRT
# pip install onnxruntime                       # CPU only, any OS
```

```python
import numpy as np, onnxruntime as ort
from PIL import Image

sess = ort.InferenceSession(
    "Cyte-V1_2x_fp16_op17_slim.onnx",
    providers=["DmlExecutionProvider", "CPUExecutionProvider"])
inp = sess.get_inputs()[0]
dt = np.float16 if "float16" in inp.type else np.float32

src = Image.open("frame.png").convert("RGB")
x = (np.asarray(src, np.float32) / 255.0).transpose(2, 0, 1)[None].astype(dt)

out = sess.run(None, {inp.name: x})[0]
Image.fromarray((out[0].transpose(1, 2, 0).clip(0, 1) * 255 + 0.5).astype("uint8")).save("frame_2x.png")
```

The `.pth` checkpoint loads as SRVGGNetCompact (24 features, 8 convolutions, 2x):

```python
import torch
from basicsr.archs.srvgg_arch import SRVGGNetCompact

model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=24,
                        num_conv=8, upscale=2, act_type='prelu')
sd = torch.load("2x-Cyte-V1-SuperUltraCompact.pth",
                map_location="cpu", weights_only=True)
model.load_state_dict(sd.get("params_ema", sd.get("params", sd)))
model.eval()
```

## License

**Dual.** [`LICENSE.MD`](LICENSE.MD) in this directory is the authoritative grant. In short:

- by default, **CC BY-NC 4.0** - the unmodified legal code is at the repository root,
  [`../LICENSE.md`](../LICENSE.md).
- **any person** using these weights as part of, through, or in connection with
  **[TheAnimeScripter](https://github.com/NevermindNilas/TheAnimeScripter)** - users
  included - has a full commercial and closed-source grant, with no further permission
  needed.

Attribution satisfying section 3(a) of the CC BY-NC 4.0 legal code can be given as:

> Cyte-V1-SuperUltraCompact by NevermindNilas -
> https://github.com/NevermindNilas/Ai-models - licensed under CC BY-NC 4.0.

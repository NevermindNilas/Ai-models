# Limbo-V1 streaming ONNX

These exports enable attention across ordered frames of one scene. They are separate from the independent-image exports in the parent directory.

| File | Size |
|---|---:|
| [Limbo_streaming_504x280_fp16.onnx](Limbo_streaming_504x280_fp16.onnx) | 50.7 MB |
| [Limbo_streaming_504x280_fp32.onnx](Limbo_streaming_504x280_fp32.onnx) | 100.6 MB |
| [Limbo_streaming_504x378_fp16.onnx](Limbo_streaming_504x378_fp16.onnx) | 50.8 MB |
| [Limbo_streaming_504x378_fp32.onnx](Limbo_streaming_504x378_fp32.onnx) | 100.7 MB |

Input `image` has shape `[N,3,H,W]`: RGB scaled to 0–1 and normalized with ImageNet mean `(0.485,0.456,0.406)` and standard deviation `(0.229,0.224,0.225)`. The first axis is **views of one scene**, not independent-image batching. Width/height are fixed per file. Output `depth` has shape `[N,H,W]` and contains positive relative depth, not display-normalized disparity. The graph does not output confidence or camera poses.

The view count is dynamic, tested from 1 through 32 in the FP16 TensorRT engines. TAS uses windows of 4/8/16/32 with 50% overlap, depth-scale alignment and overlap blending. That scheduling/postprocessing is outside ONNX. Send a partial final window at its real length; duplicate padding changes attention. The reference view is the first frame.

FP16 is mixed precision: attention matrix products, normalization, softmax, resize and exponentials retain FP32 for stability. FP32 exports use float32 throughout. Opset 17. Use TensorRT optimization profiles with minimum N=1 and maximum N equal to the intended window; build engines on the target machine. No machine-specific engines are shipped here. These multi-view exports were validated for TensorRT; the parent directory's DirectML performance claims do not apply.

In TheAnimeScripter, select `video_limbo-tensorrt` and `--depth_window 8`; FP16 is the default (`--half True`). TAS at the exporter commit still generates/caches the graph locally from safetensors; publishing these files does not change that downloader behavior.

See the [validation report](https://github.com/NevermindNilas/TheAnimeScripter/blob/f2f34c9c57d44c81051744576a71979c7d8c8976/docs/LIMBO_TENSORRT_VALIDATION.md), [exporter](https://github.com/NevermindNilas/TheAnimeScripter/blob/f2f34c9c57d44c81051744576a71979c7d8c8976/src/depth/streaming_export.py), and [manifest](manifest.json) for provenance and checksums. Source safetensors match this model directory's published checkpoint byte-for-byte.

The parent [license](../LICENSE.MD) and [Apache-2.0 notices](../LICENSE-APACHE-2.0.MD) apply. These are derived exports of the same checkpoint; no training or license change is introduced.

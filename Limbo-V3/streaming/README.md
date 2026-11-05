# Limbo-V3 streaming ONNX

These exports enable attention across ordered frames of one scene. They are separate from the independent-image exports in the parent directory.

| File | Size |
|---|---:|
| [Limbo_streaming_504x280_fp16.onnx](Limbo_streaming_504x280_fp16.onnx) | 50.7 MB |
| [Limbo_streaming_504x280_fp32.onnx](Limbo_streaming_504x280_fp32.onnx) | 100.6 MB |
| [Limbo_streaming_504x378_fp16.onnx](Limbo_streaming_504x378_fp16.onnx) | 50.8 MB |
| [Limbo_streaming_504x378_fp32.onnx](Limbo_streaming_504x378_fp32.onnx) | 100.7 MB |

Input `image` has shape `[N,3,H,W]`: RGB scaled to 0–1 and normalized with ImageNet mean `(0.485,0.456,0.406)` and standard deviation `(0.229,0.224,0.225)`. The first axis is **views of one scene**, not independent-image batching. Width/height are fixed per file. Output `depth` has shape `[N,H,W]` and contains positive relative depth, not display-normalized disparity. The graph does not output confidence or camera poses.

The view count is dynamic. TAS uses windows of 4/8/16/32 with 50% overlap, depth-scale alignment and overlap blending. That scheduling/postprocessing is outside ONNX. Send a partial final window at its real length; duplicate padding changes attention. The reference view is the first frame.

FP16 is mixed precision: attention matrix products, normalization, softmax, resize and exponentials retain FP32 for stability. FP32 exports use float32 throughout. Opset 17. Use TensorRT optimization profiles with minimum N=1 and maximum N equal to the intended window; build engines on the target machine. No machine-specific engines are shipped here.

These graphs are the same topology as Limbo-V2's streaming files (identical byte sizes and exporter), with Limbo-V3 weights. They were checked against the eager PyTorch multi-view model through onnxruntime (CPU) at 1, 3 and 8 views: relative MAE ≤ 1.8e-06 for FP32 and ≤ 1.0e-03 for FP16 (per-file numbers in the [manifest](manifest.json)). They have **not** yet been re-validated under TensorRT; Limbo-V2's [TensorRT validation](https://github.com/NevermindNilas/TheAnimeScripter/blob/f2f34c9c57d44c81051744576a71979c7d8c8976/docs/LIMBO_TENSORRT_VALIDATION.md) covers the same graph structure. Limbo-V3's video-flicker gains (parent README) were measured in exactly this streaming mode (16-frame windows).

Built with TheAnimeScripter's [exporter](https://github.com/NevermindNilas/TheAnimeScripter/blob/7b9d8835415c27c2b60067ba72dbf87d1d584868/src/depth/streaming_export.py) at `7b9d8835`, whose graph code is identical to `f2f34c9c` (the commit that built Limbo-V1/V2's streaming files). Source safetensors match this model directory's published checkpoint byte-for-byte (`source_checkpoint_sha256` in the manifest). TheAnimeScripter does not ship a Limbo-V3 depth method yet; until it does, these files are for direct use.

The parent [license](../LICENSE.MD) and [Apache-2.0 notices](../LICENSE-APACHE-2.0.MD) apply. These are derived exports of the same checkpoint; no training or license change is introduced.

# Limbo-V1 — License

Copyright (c) 2026 NevermindNilas.

## 1. Definitions

**The Files** — the model weights published as Limbo-V1 in this directory:

| file |
| :-- |
| `Limbo_504x280_fp32.onnx` |
| `Limbo_504x280_fp16.onnx` |
| `Limbo_504x280_b4_fp16.onnx` |
| `Limbo_504x378_fp32.onnx` |
| `Limbo_504x378_fp16.onnx` |
| `Limbo.safetensors` |

These are the names as published here. Where the same bytes are re-hosted or mirrored under a
different name, these terms follow the bytes.

**The Upstream Material** — the tensor values, architecture and graph structure originating in
Depth Anything 3 (section 4), which are present in the Files. Roughly a quarter of the
checkpoint's parameters are upstream values carried through unchanged.

**The Contribution** — everything in the Files that is NevermindNilas's own work: the
fine-tuned tensor values, the weight averaging, and the ONNX exports. Sections 2 and 3 license
the Contribution. They say nothing about, and cannot restrict, your rights in the Upstream
Material, which reach you directly under the Apache License 2.0.

## 2. Public license — CC BY-NC 4.0

The Contribution is licensed to the public under the **Creative Commons
Attribution-NonCommercial 4.0 International** license (CC BY-NC 4.0). The legal code applies
unmodified; it is at [`../LICENSE.md`](../LICENSE.md) in this repository and at
<https://creativecommons.org/licenses/by-nc/4.0/legalcode>. Section 3 below is an *additional*
grant; holding it neither replaces nor limits anyone's rights under this section, and you may
rely on either or both.

Plain-language summary, for orientation only and not a substitute for the legal code: you may
use, share and adapt the Contribution for non-commercial purposes, with attribution and an
indication of changes. Commercial use is not granted here — except under section 3. If your use
sits near that line and section 3 does not cover you, treat it as commercial and ask (section
6).

Attribution satisfying section 3(a) of the legal code can be given as:

> Limbo-V1 by NevermindNilas — https://github.com/NevermindNilas/Ai-models — licensed under
> CC BY-NC 4.0. Derived from Depth Anything 3 SMALL (Apache-2.0); modified — see the NOTICE in
> section 4 of the accompanying LICENSE.MD.

**Model outputs.** NevermindNilas asserts no copyright in the depth maps, disparity maps or
other inference outputs produced by running the Files, and does not treat them as Adapted
Material for the purposes of this license. Producing outputs is not by itself commercial use;
what you do with them may be.

## 3. Additional grant — use through TheAnimeScripter

To the full extent of the rights NevermindNilas holds in the Files, NevermindNilas grants to
**any person**, in respect of their use of the Files as part of, through, or in connection with
**TheAnimeScripter**, a perpetual, worldwide, irrevocable, royalty-free, non-exclusive and
sublicensable license to use, reproduce, modify, convert, re-export, embed, distribute, sell and
otherwise commercially exploit the Files — including in closed-source, proprietary, paid and
binary-only distributions.

This grant is for **users** of TheAnimeScripter as much as for its author and anyone who
distributes it. If you run the Files through TheAnimeScripter, your use — including commercial
use, and including work you sell or deliver to clients — is covered by this section, with no
further permission needed and no distinction drawn between users, contributors and maintainers.

**TheAnimeScripter** means the software currently published at
<https://github.com/NevermindNilas/TheAnimeScripter> and any successor, renamed or rewritten
version of it maintained by NevermindNilas, including its releases, bundled builds and
installers.

**Extent of the waiver.** No attribution, non-commercial, share-alike or source-disclosure
condition *imposed by NevermindNilas* applies to this grant. This waiver reaches only conditions
NevermindNilas is entitled to impose. It does not waive, and cannot waive, the Apache License
2.0 obligations attaching to the Upstream Material: the notices in section 4 must still travel
with any redistribution of the Files, including closed-source ones. In practice that is one
short notice file, not a source-disclosure obligation — and it binds someone shipping the Files
onward, not someone merely running the software.

**Scope.** This grant covers the Files only. It does not extend to evaluation imagery,
comparison figures or other material published elsewhere.

**No representation as to training data.** The Files were produced by training on third-party
image corpora. Some of those corpora declare no license, and one photographic set used as a
training-time anchor derives from a database distributed for research purposes. NevermindNilas
makes no representation or warranty that the Files are free of third-party rights arising from
that training data, and this grant conveys only NevermindNilas's own rights (see also section
5).

## 4. Upstream material, notices and changes

The Files are a **derivative work** of Apache-2.0 licensed material:

| Upstream | Role |
| :-- | :-- |
| [`depth-anything/DA3-SMALL`](https://huggingface.co/depth-anything/DA3-SMALL) | pretrained weights that were fine-tuned — its values are present in the Files |
| [`ByteDance-Seed/Depth-Anything-3`](https://github.com/ByteDance-Seed/Depth-Anything-3) | model code defining the topology and the exported op sequence |

### NOTICE — copyright notices retained under Apache License 2.0 §4(c)

```
Copyright 2025 The Depth Anything 3 Team
Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
Portions Copyright (c) Meta Platforms, Inc. and affiliates  (DINOv2 backbone code,
  itself derived from facebookresearch/dino and rwightman/pytorch-image-models)
```

That material remains licensed under the Apache License, Version 2.0. A verbatim copy of the
license as published by the Depth Anything 3 authors is included here as
[`LICENSE-APACHE-2.0.MD`](LICENSE-APACHE-2.0.MD), satisfying §4(a), and is also at
<http://www.apache.org/licenses/LICENSE-2.0>. The upstream release ships no separate `NOTICE`
file, so no upstream `NOTICE` text is carried forward beyond the block above.

### Statement of changes (Apache License 2.0 §4(b))

The Files are a **modified** derivative of `depth-anything/DA3-SMALL`:

1. The DINOv2 backbone and the DPT depth head were fine-tuned on an anime image corpus by
   knowledge distillation, with a function-space anchor on the pretrained model's own
   predictions to bound catastrophic forgetting.
2. The camera encoder, camera decoder and auxiliary ray head received no gradients; their values
   are the upstream values up to exponential-moving-average float rounding (at most 1.3e-05
   absolute). Six auxiliary-head tensors
   (`head.scratch.output_conv2_aux.{1,2,3}.2.{weight,bias}`) do not exist in the upstream
   release, were never trained, and remain at random initialisation in the checkpoint.
3. The published checkpoint is a weighted average (0.75 / 0.25) of two such fine-tuning runs,
   stored at fp16.
4. The ONNX artifacts export the monocular depth path only: the camera-pose branch and the
   auxiliary ray head were removed from the exported graph, and the single-view axis was fixed
   inside it. Resolution and batch size are static per file.

### Trademarks

No trademark rights are granted, per Apache License 2.0 §6. "Depth Anything", "ByteDance" and
"Meta" are marks of their respective owners. Limbo-V1 is an independent derivative work and is
**not** affiliated with, endorsed by, or a product of those parties.

## 5. No warranty

The Files are provided on an **"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND**,
either express or implied, including without limitation any warranty of merchantability, fitness
for a particular purpose, non-infringement, or freedom from third-party rights arising from the
training data. In no event shall NevermindNilas be liable for any claim, damages or other
liability arising from, out of, or in connection with the Files or their use. The upstream
licensors' own disclaimers and limitations under Apache License 2.0 §§7–8 apply to the Upstream
Material and reach you directly; nothing here is stated on their behalf.

The Files predict *relative* depth from stylised animation. They are not suitable for metric
measurement, robotics, medical use, or any application in which an incorrect depth estimate can
cause harm.

## 6. Commercial licensing

Commercial use through TheAnimeScripter is already granted by section 3. For commercial terms
outside it, terms may be available at NevermindNilas's discretion; no offer is made by this
file. Contact <https://github.com/NevermindNilas>.

## 7. Severability

If any provision of this file is held unenforceable, it is severed and the remainder continues in
effect. Severance of any part of section 3 does not affect section 2, and vice versa.

---

*This file is a license grant written by the author of the Files. It is not legal advice, and it
does not state a governing law — add one if you need it enforced in a particular forum.*

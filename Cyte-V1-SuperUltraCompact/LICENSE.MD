# Cyte-V1-SuperUltraCompact - License

Copyright (c) 2026 NevermindNilas.

## 1. Definitions

**The Files** - the model weights published as Cyte-V1-SuperUltraCompact in this directory:

| file |
| :-- |
| `2x-Cyte-V1-SuperUltraCompact.pth` |
| `Cyte-V1_2x_fp16_op17_slim.onnx` |
| `Cyte-V1_2x_fp32_op17_slim.onnx` |

These are the names as published here. Where the same bytes are re-hosted or mirrored under a
different name, these terms follow the bytes.

**The Architecture** - the SRVGGNetCompact network topology ("SuperUltraCompact", 24 features,
8 convolutions, 2x) originating in Real-ESRGAN (section 4). The Architecture's code is not
distributed here; only its shape is reflected in the Files.

**The Contribution** - the weights in the Files. Sections 2 and 3 license the Contribution.

## 2. Public license - CC BY-NC 4.0

The Contribution is licensed to the public under the **Creative Commons
Attribution-NonCommercial 4.0 International** license (CC BY-NC 4.0). The legal code applies
unmodified; it is at [`../LICENSE.md`](../LICENSE.md) in this repository and at
<https://creativecommons.org/licenses/by-nc/4.0/legalcode>. Section 3 below is an *additional*
grant; holding it neither replaces nor limits anyone's rights under this section, and you may
rely on either or both.

Plain-language summary, for orientation only and not a substitute for the legal code: you may
use, share and adapt the Contribution for non-commercial purposes, with attribution and an
indication of changes. Commercial use is not granted here - except under section 3. If your use
sits near that line and section 3 does not cover you, treat it as commercial and ask (section
6).

Attribution satisfying section 3(a) of the legal code can be given as:

> Cyte-V1-SuperUltraCompact by NevermindNilas -
> https://github.com/NevermindNilas/Ai-models - licensed under CC BY-NC 4.0.

**Model outputs.** NevermindNilas asserts no copyright in the upscaled images or other
inference outputs produced by running the Files, and does not treat them as Adapted Material
for the purposes of this license. Producing outputs is not by itself commercial use; what you
do with them may be.

## 3. Additional grant - use through TheAnimeScripter

To the full extent of the rights NevermindNilas holds in the Files, NevermindNilas grants to
**any person**, in respect of their use of the Files as part of, through, or in connection with
**TheAnimeScripter**, a perpetual, worldwide, irrevocable, royalty-free, non-exclusive and
sublicensable license to use, reproduce, modify, convert, re-export, embed, distribute, sell and
otherwise commercially exploit the Files - including in closed-source, proprietary, paid and
binary-only distributions.

This grant is for **users** of TheAnimeScripter as much as for its author and anyone who
distributes it. If you run the Files through TheAnimeScripter, your use - including commercial
use, and including work you sell or deliver to clients - is covered by this section, with no
further permission needed and no distinction drawn between users, contributors and maintainers.

**TheAnimeScripter** means the software currently published at
<https://github.com/NevermindNilas/TheAnimeScripter> and any successor, renamed or rewritten
version of it maintained by NevermindNilas, including its releases, bundled builds and
installers.

**Extent of the waiver.** No attribution, non-commercial, share-alike or source-disclosure
condition *imposed by NevermindNilas* applies to this grant.

**Scope.** This grant covers the Files only. It does not extend to evaluation imagery,
comparison figures or other material published elsewhere.

**No representation as to training data.** The Files were produced by training on third-party
image corpora (section 4). Some of those corpora declare no license. NevermindNilas makes no
representation or warranty that the Files are free of third-party rights arising from that
training data, and this grant conveys only NevermindNilas's own rights (see also section 5).

## 4. Architecture, provenance and training

The Files implement the Real-ESRGAN Compact architecture
([`xinntao/Real-ESRGAN`](https://github.com/xinntao/Real-ESRGAN), SRVGGNetCompact,
BSD-3-Clause, copyright (c) 2021 Xintao Wang). A verbatim copy of that license is included
here as [`LICENSE-BSD-3-CLAUSE.MD`](LICENSE-BSD-3-CLAUSE.MD) for reference. No third-party
model weights are contained in the Files.

| Training detail | Value |
| :-- | :-- |
| Architecture | SRVGGNetCompact ("SuperUltraCompact"), 24nf / 8nc, 2x, RGB - 45,156 parameters |
| Datasets | [`Zarxrax/ModernAnimation1080_v3`](https://huggingface.co/datasets/Zarxrax/ModernAnimation1080_v3), [`umzi/DigitalArtV3`](https://huggingface.co/datasets/umzi/DigitalArtV3), [`umzi/DigitalArt_v5`](https://huggingface.co/datasets/umzi/DigitalArt_v5) |
| Iterations | Undisclosed |
| OTF (on-the-fly degradations) | Yes |

### Trademarks

No trademark rights are granted. "Cyte-V1-SuperUltraCompact" is an independent work and is
**not** affiliated with or endorsed by the authors of the Architecture or the datasets.

## 5. No warranty

The Files are provided on an **"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND**,
either express or implied, including without limitation any warranty of merchantability, fitness
for a particular purpose, non-infringement, or freedom from third-party rights arising from the
training data. In no event shall NevermindNilas be liable for any claim, damages or other
liability arising from, out of, or in connection with the Files or their use.

The Files are an ultra-compact stylised-animation upscaler. They trade capacity for speed and
are not suitable for any application in which an incorrect upscaled result can cause harm.

## 6. Commercial licensing

Commercial use through TheAnimeScripter is already granted by section 3. For commercial terms
outside it, terms may be available at NevermindNilas's discretion; no offer is made by this
file. Contact <https://github.com/NevermindNilas>.

## 7. Severability

If any provision of this file is held unenforceable, it is severed and the remainder continues in
effect. Severance of any part of section 3 does not affect section 2, and vice versa.

---

*This file is a license grant written by the author of the Files. It is not legal advice, and it
does not state a governing law - add one if you need it enforced in a particular forum.*

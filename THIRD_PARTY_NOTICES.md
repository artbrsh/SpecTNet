# Third-party notices

[SpecTNet](https://github.com/artbrsh/SpecTNet.git): [MIT](LICENSE).

| Component | License |
|:----------|:--------|
| [Swin Transformer](https://github.com/microsoft/swin-transformer) | [MIT](LICENSE) |
| [TinyViT](https://github.com/microsoft/Cream/tree/main/TinyViT) | [MIT](LICENSE) |
| [LeViT](https://github.com/facebookresearch/LeViT) | [Apache-2.0](licenses/LeViT-Apache-2.0.txt) |
| [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) | Apache-2.0 |
| [PyTorch](https://github.com/pytorch/pytorch) | BSD-style |

LeViT → TinyViT → SpecTNet.

- Source: [LeViT/levit.py](https://github.com/facebookresearch/LeViT/blob/main/levit.py).
- Via: [TinyViT/models/tiny_vit.py](https://github.com/microsoft/Cream/blob/main/TinyViT/models/tiny_vit.py).
- LeViT license: [Apache-2.0](licenses/LeViT-Apache-2.0.txt).
- TinyViT: Copyright (c) 2022 Microsoft. [MIT](LICENSE).

Portions modified in SpecTNet:

| File | Portions derived from LeViT |
|:-----|:---------------------------|
| [models/low_level.py](models/low_level.py) | `Conv2d_BN`, including `fuse()` |
| [models/Attention.py](models/Attention.py) | Relative-position biases, their indices and evaluation cache |

LeViT source notices:

```text
Copyright (c) 2015-present, Facebook, Inc.
All rights reserved.

Modified from
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
Copyright 2020 Ross Wightman, Apache-2.0 License
```

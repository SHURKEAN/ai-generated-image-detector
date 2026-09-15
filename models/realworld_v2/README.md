# Real-world detector v2

This directory is deliberately separate from the published CTDS checkpoints.

- `upstream/` contains locally cached, checksum-verifiable backbone assets.
- `realworld_v2_best.pth` contains only the locally trained academic binary
  head and its calibration metadata. It is an experiment, not the current
  automatic application route.
- `runs/` contains training history and evaluation reports.

The bundled UniversalFakeDetect head is an **academic reference baseline**. It
must not be confused with a product-safe model trained from the project's own
licensed manifest.

## 2026-09-06 academic run

The first source-held-out run used 5,931 training images and 2,306 validation
images from CommunityForensics-Small. The best checkpoint was epoch 1:

- held-out balanced accuracy: 79.66%
- genuine-image false-positive rate: 4.94%
- external smoke test: the private phone photo was correct, but only 1 of 5
  official DALL-E samples was detected

It therefore failed promotion. The app's `auto` mode continues to use the
Community Forensics CVPR 2025 checkpoint, which scored 99.83% balanced accuracy
on the same 2,306-image validation split and 6/6 on the small external smoke
set. These are project measurements, not claims of universal accuracy.

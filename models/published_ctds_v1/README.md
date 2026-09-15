# Published CTDS baseline v1.0

This directory protects the identity of the model reported in the published
CTDS paper. The large checkpoint files remain in `../tuned/`; `manifest.json`
records their immutable paths, byte sizes and SHA-256 checksums.

Do not train into, rename, or overwrite the four referenced checkpoint files.
All post-publication work belongs under `models/realworld_v2/` and must be
described as a real-world extension rather than as part of the published result.

Verify the baseline at any time with:

```powershell
.\.venv-tuned\Scripts\python.exe scripts\verify_published_baseline.py
```


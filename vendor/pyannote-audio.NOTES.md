# pyannote.audio vendor note

Full repo: https://github.com/pyannote/pyannote-audio (large).

Runtime: `pip install pyannote.audio`.

Default weight source: **ModelScope** (no HF token required):
- https://modelscope.cn/models/pyannote/speaker-diarization-3.1
- Search ModelScope for `pyannote segmentation` / `segmentation-3.0`

Local cache: `models/pyannote/`

```bash
python -m vadbench.download --engine pyannote
```

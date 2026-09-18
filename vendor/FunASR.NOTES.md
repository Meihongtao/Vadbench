# FunASR vendor note

Full repo: https://github.com/modelscope/FunASR (large).

This project does not vendor the entire toolkit. Runtime uses `pip install funasr`.
VAD model weights are stored under `models/funasr_fsmn/`.

Official standalone usage:

```python
from funasr import AutoModel
model = AutoModel(model="fsmn-vad", disable_update=True)
print(model.generate(input="audio.wav")[0]["value"])
```

VAD model sources:
- https://huggingface.co/funasr/fsmn-vad
- ModelScope: damo/speech_fsmn_vad_zh-cn-16k-common-pytorch

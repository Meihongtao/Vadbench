# Vadbench 开源 VAD 模型说明

本文档记录本仓库接入的五家 VAD：官方链接、实现、模型下载与本地归档位置。
统一调用接口见 Python 包 `vadbench`（段级时间戳，单位：秒）。

评测结果与选型建议见 [README](../README.md)；评测协议见 [eval.md](eval.md)。

## 环境（uv + 阿里云镜像）

先退出 conda，再用 uv 创建独立 Python 3.11 虚拟环境（勿用 conda base）：

```powershell
conda deactivate
# 如仍在 conda 环境中，可再执行一次 conda deactivate

# 项目已配置 [[tool.uv.index]] 指向阿里云；安装时显式指定更稳妥：
$env:UV_INDEX_URL = "https://mirrors.aliyun.com/pypi/simple/"
uv python install 3.11
uv venv --python 3.11 .venv
.\.venv\Scripts\Activate.ps1

# 若本机 uv pip 改写 Windows 入口报错，改用 venv 内 pip + 阿里云：
python -m pip install -U pip setuptools wheel -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
python -m pip install -e ".[webrtc,silero,funasr,firered,download]" -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
python -m pip install -e vendor\FireRedVAD -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

```python
from vadbench import create

vad = create("silero")  # webrtc | silero | funasr_fsmn | pyannote | firered
segments = vad.detect("audio.wav")
for s in segments:
    print(s.start, s.end)
```

下载模型与归档官方源码：

```powershell
# 若直连 Hugging Face 失败：
$env:HF_ENDPOINT = "https://hf-mirror.com"
python -m vadbench.download --engine all
# 或按引擎: webrtc silero funasr_fsmn pyannote firered
```

权重目录：`models/<engine>/`  
源码归档：`vendor/<repo>/`（只读参考；推理优先走 pip 包）

---

## 1. WebRTC VAD

| 项 | 内容 |
|----|------|
| 官方 Python 封装 | https://github.com/wiseman/py-webrtcvad |
| 上游算法 | Google WebRTC VAD（C，随 py-webrtcvad 内嵌） |
| PyPI | `webrtcvad` |
| 模型权重 | **无独立权重文件**（算法内嵌） |
| 本地 vendor | `vendor/py-webrtcvad` |
| 采样率 | 8 / 16 / 32 / 48 kHz；本仓库统一 16 kHz |
| 帧长 | 10 / 20 / 30 ms |
| 许可证 | MIT（封装）；WebRTC 相关见上游 |

```bash
pip install webrtcvad
python -m vadbench.download --engine webrtc   # 仅归档源码
```

---

## 2. Silero VAD

| 项 | 内容 |
|----|------|
| 官方仓库 | https://github.com/snakers4/silero-vad |
| PyPI | `silero-vad` |
| 模型 | 仓库内 ONNX / JIT（建议 ONNX） |
| 本地 models | `models/silero/silero_vad.onnx` |
| 本地 vendor | `vendor/silero-vad` |
| 采样率 | 8 / 16 kHz（本仓库 16 kHz） |
| 许可证 | MIT |

```bash
pip install silero-vad torch torchaudio
python -m vadbench.download --engine silero
```

Wiki（版本与模型）：https://github.com/snakers4/silero-vad/wiki/Version-history-and-Available-Models

---

## 3. FunASR-FSMN VAD

| 项 | 内容 |
|----|------|
| 官方工具包 | https://github.com/modelscope/FunASR |
| 模型 (HF) | https://huggingface.co/funasr/fsmn-vad |
| 模型 (ModelScope) | `damo/speech_fsmn_vad_zh-cn-16k-common-pytorch` / `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` |
| ONNX 变体 | https://huggingface.co/funasr/fsmn-vad-onnx |
| 本地 models | `models/funasr_fsmn/` |
| 本地 vendor | `vendor/FunASR`（sparse：仅 VAD 相关说明与最小引用；完整仓过大） |
| 采样率 | 16 kHz |
| 原生输出 | 毫秒段 `[[start_ms, end_ms], ...]` → 本仓库转为秒 |
| 许可证 | FunASR MIT；模型见 MODEL_LICENSE |

```bash
pip install funasr torch
python -m vadbench.download --engine funasr_fsmn
```

独立调用示例（官方）：

```python
from funasr import AutoModel
model = AutoModel(model="fsmn-vad", disable_update=True)
print(model.generate(input="audio.wav")[0]["value"])
```

---

## 4. Pyannote VAD

| 项 | 内容 |
|----|------|
| 官方工具包 | https://github.com/pyannote/pyannote-audio |
| ModelScope（默认下载） | https://modelscope.cn/models/pyannote/speaker-diarization-3.1 ；可搜索 `pyannote segmentation` |
| VAD Pipeline (HF 备选) | https://huggingface.co/pyannote/voice-activity-detection |
| Segmentation 骨干 | https://huggingface.co/pyannote/segmentation-3.0 |
| 本地 models | `models/pyannote/segmentation-3.0` 或 `speaker-diarization-3.1` |
| 本地 vendor | 不全量 clone；见 `vendor/pyannote-audio.NOTES.md` |
| 认证 | **优先 ModelScope，无需 HF_TOKEN**；仅 ModelScope 失败时才回退 HF |
| 采样率 | 通常 16 kHz |
| 许可证 | MIT（代码）；模型条款见托管站 |

```powershell
pip install pyannote.audio torch modelscope
python -m vadbench.download --engine pyannote
```

评测协议见 [docs/eval.md](eval.md)。

---

## 5. FireRedVAD（小红书 FireRedTeam）

| 项 | 内容 |
|----|------|
| 官方仓库 | https://github.com/FireRedTeam/FireRedVAD |
| 模型 (HF) | https://huggingface.co/FireRedTeam/FireRedVAD |
| 模型 (ModelScope) | `xukaituo/FireRedVAD` |
| 本地 models | `models/firered/VAD/`（非流式） |
| 本地 vendor | `vendor/FireRedVAD` |
| 采样率 | 16 kHz |
| 本仓库用法 | 非流式 `FireRedVad.detect` → `timestamps` |
| 许可证 | 见官方仓库 LICENSE |

```bash
pip install torch torchaudio
python -m vadbench.download --engine firered
pip install -e vendor/FireRedVAD
# 或依赖 vendor 路径自动加入 sys.path（见 backends/firered.py）
```

---

## 统一接口约定

| 约定 | 说明 |
|------|------|
| 工厂 | `vadbench.create(name, **kwargs)` |
| 名称 | `webrtc` / `silero` / `funasr_fsmn` / `pyannote` / `firered` |
| 输入 | wav 路径或 float32 mono 波形 |
| 内部 | 重采样到 16 kHz mono |
| 输出 | `list[SpeechSegment(start, end)]`，单位**秒** |
| 不做 | 帧级概率 API（多数引擎不统一开放，不利于公平评测） |

## 归档 commit 记录

下载完成后，`python -m vadbench.download` 会把各 vendor 的 commit / 模型 revision 写入 `models/DOWNLOAD_MANIFEST.json`。

# VAD 中英双语评测协议

主文档见 [README](../README.md)，完整指标表见 [实验结果汇总.md](实验结果汇总.md)。

## 数据（下载到仓库 `data/`）

| 代号 | 语言 | 数据集 | 本地目录 |
|------|------|--------|----------|
| `aishell4` | 中 | AISHELL-4 test | `data/aishell4_test/` |
| `alimeeting` | 中 | AliMeeting test（远场 ch0） | `data/alimeeting_far_test/` |
| `alimeeting_eval` | 中 | AliMeeting eval（远场 ch0，国内 OSS） | `data/alimeeting_far_eval/` |
| `ami` | 英 | AMI IHM test | `data/ami_ihm_test/` |
| `voxconverse` | 英 | VoxConverse test（牛津源，国内常很慢） | `data/voxconverse_test/` |

下载源（优先国内 / 非 HF）：

- **AliMeeting test/eval**：ModelScope 元数据 + 阿里云 OSS；test 用 pyannote RTTM，eval 用 TextGrid 转 RTTM
- **VoxConverse**：官方 Oxford（可选；国内慢时请用 `alimeeting_eval`）

```powershell
.\.venv\Scripts\Activate.ps1
python -m vadbench.eval.download_data --dataset alimeeting,alimeeting_eval
```

GT：各说话人时间段**并集** → speech / non-speech。

## 指标（10 ms 帧）

主指标 **F1**；同时报告 Precision、Recall、FAR、Miss Rate（时长加权汇总）。

不做 AUC（各引擎段级输出不统一暴露帧概率）。

## 跑评测

```
python -m vadbench.eval --engines webrtc,silero,funasr_fsmn,firered,pyannote --dataset alimeeting,voxconverse --device cpu
# 冒烟：
python -m vadbench.eval --engines webrtc,silero --dataset alimeeting,voxconverse --max-files 1
```

结果：`results/eval_*.json`、`results/summary_*.csv`。

二次分析（每文件分布、FAR 分母口径、跨数据集透视，不重跑模型）：

```
python -m vadbench.report
```

## Pyannote 权重

优先 ModelScope → `models/pyannote/`：

```powershell
python -m vadbench.download --engine pyannote
```

ModelScope 参考：https://modelscope.cn/models/pyannote/speaker-diarization-3.1 （并可搜索 segmentation）。

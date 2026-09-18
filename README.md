# Vadbench

五个开源 VAD 引擎的统一段级评测框架：**5 个模型 × 5 个数据集 × 57 小时中文/英文会议与对话音频**。

> 📝 评测笔记（结论、每文件分布、选型建议）：<https://meihongtao.github.io/blogs/Vad%E8%AF%84%E6%B5%8B.html>

本仓库只做评测，不含任何 VAD 实现。所有引擎通过统一接口接入，保证对比的是同一件事。

## 评测对象

| 引擎 | 类型 | 权重 | 上游 |
|------|------|-----:|------|
| `webrtc` | 传统能量/频谱 | 无独立权重 | [wiseman/py-webrtcvad](https://github.com/wiseman/py-webrtcvad) |
| `silero` | 轻量 DNN | 2.22 MB | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) |
| `funasr_fsmn` | FSMN | 1.65 MB | [modelscope/FunASR](https://github.com/modelscope/FunASR) |
| `firered` | mel + 检测网络 | 2.28 MB | [FireRedTeam/FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) |
| `pyannote` | 端到端分割 | 5.64 MB | [pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio) |

## 评测数据

| 数据集 | 语言 / 场景 | 规模 | 标注 |
|--------|-------------|------|------|
| AliMeeting far test | 中 / 会议远场 | 20 场 · 10.78 h | RTTM |
| AliMeeting far eval | 中 / 会议远场 | 8 场 · 4.21 h | TextGrid → RTTM |
| AISHELL-4 test | 中 / 会议（麦克风阵列） | 20 场 · 12.73 h | RTTM |
| MagicData-RAMC test | 中 / 手机对话近场 | 43 场 · 20.64 h | 语音活动时间戳 → RTTM |
| AMI IHM test | 英 / 会议近场 | 16 场 · 9.06 h | 段级时间戳 |

合计 **107 场 · 57.4 小时**。GT 统一口径为各说话人时间段的**并集**，即 speech / non-speech 二分类。

## 结果

### 时长加权 F1

| 引擎 | AliMeeting test | AliMeeting eval | AISHELL-4 | MagicData-RAMC | AMI IHM |
|------|----------------:|----------------:|----------:|---------------:|--------:|
| **pyannote** | **0.993** | **0.980** | **0.980** | 0.940 | **0.969** |
| **firered** | 0.954 | 0.950 | 0.933 | **0.947** | 0.917 |
| **funasr_fsmn** | 0.948 | 0.933 | 0.937 | 0.940 | 0.947 |
| **silero** | 0.889 | 0.867 | 0.868 | 0.941 | 0.918 |
| **webrtc** | 0.880 | 0.864 | 0.746 | 0.924 | 0.902 |

指标定义与完整的 P / R / FAR / Miss 明细见 [`docs/实验结果汇总.md`](docs/实验结果汇总.md)。

⚠️ 报 F1 必须写明聚合口径。上表是时长加权（先求和各文件 TP/FP/TN/FN 再算 F1）。换成文件平均，AliMeeting test 上 silero 与 webrtc 的胜负会互换 —— 因为 silero 有单个文件 F1 只有 0.074，文件平均被离群值拖死，而时长加权把它稀释掉了。

### 每文件分布（均值掩盖了什么）

`python -m vadbench.report` 会给出最差值 / 中位数 / 均值 / F1 < 0.9 的文件数：

```
dataset               engine         files     min     p25  median    mean     max   <0.9
-----------------------------------------------------------------------------------------
AliMeeting test       pyannote          20   0.983   0.989   0.994   0.993   0.998      0
AliMeeting test       silero            20   0.074   0.887   0.941   0.856   0.981      6
AliMeeting test       webrtc            20   0.370   0.884   0.928   0.858   0.974      5
...
AISHELL-4             webrtc            20   0.510   0.599   0.718   0.729   0.915     17
```

pyannote 是唯一「没有差文件」的引擎。silero / webrtc 的问题不是普遍差，而是个别文件直接崩 —— 同一批数据里既是「还不错」又是「完全不可用」。

### FAR 的分母陷阱

会议音频里语音帧占比 80%~92%，非语音帧是少数派。分母一小，FAR 就被放大：

| 数据集 | 引擎 | FAR | FP/N | 读法 |
|--------|------|----:|-----:|------|
| MagicData-RAMC | funasr_fsmn | 0.600 | 10.30% | 每 10 帧有 1 帧非语音进 ASR |
| AliMeeting test | funasr_fsmn | 0.429 | 3.24% | 每 30 帧有 1 帧 |
| AISHELL-4 | silero | 0.035 | 0.33% | 可忽略 |

`FAR = FP/(FP+TN)` 是标准 FPR；`FP/N` 才是「有多少帧垃圾被送进下游」。建议同时报，或只报 Precision。

### ONNX 端到端延迟

条件：10 s / 16 kHz 单通道测音，warmup 5 + 正式 50 次，`intra_op_num_threads = inter_op_num_threads`，延迟含特征提取 + 推理 + 后处理成段（不含模型加载）。

| 引擎 | format | 体积 MB | mean@T1 | mean@T2 | mean@T4 | RTF@T2 | p95@T2 |
|------|--------|--------:|--------:|--------:|--------:|-------:|-------:|
| webrtc | native | n/a | 2.3 ms | — | — | **0.00023** | 3.4 ms |
| firered | onnx | 2.28 | 33.2 ms | 26.8 ms | 25.5 ms | 0.00268 | 30.6 ms |
| pyannote | onnx | 5.64 | 37.4 ms | 27.3 ms | 25.4 ms | 0.00273 | 30.7 ms |
| silero | onnx | 2.22 | 64.8 ms | 59.7 ms | 54.9 ms | 0.00597 | 67.0 ms |
| funasr_fsmn | onnx | 1.65 | 121.2 ms | 118.4 ms | 118.8 ms | 0.01184 | 124.7 ms |

**先证明「是同一个模型」，再比快慢。** 跨后端比性能有个隐蔽前提：ONNX 导出会带来数值漂移，如果漂移改变了段边界，比的就是两个不同的东西。所以 `vadbench/perf/align.py` 在跑性能前先用原始后端和 ONNX 后端各推理一次、栅格化到 10 ms 帧算 F1，`F1 >= 0.99` 才算对齐通过，否则该引擎的性能数据标 `fail_align` 丢弃。本次四个模型全部是 **align F1 = 1.000**，即段级输出逐帧完全一致。

## 安装

需要 Python 3.10–3.12。推荐 uv：

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1

# 只装框架 + WebRTC
pip install -e ".[webrtc]"
# 全部五个引擎
pip install -e ".[all]"
```

各引擎的额外依赖见 `pyproject.toml` 的 `[project.optional-dependencies]`。FireRedVAD 若要跑原始（非 ONNX）后端，还需 `pip install -e vendor/FireRedVAD`；本仓库未内置该 vendor 目录，见下文「模型与数据」。

下载权重（写入 `models/`）与归档上游源码（写入 `vendor/`）：

```bash
python -m vadbench.download --engine all
# 或按引擎：webrtc silero funasr_fsmn pyannote firered
```

## 用法

### 统一接口

输入 wav 路径或 float32 单通道波形，内部统一重采样到 16 kHz 单通道；输出 `list[SpeechSegment(start, end)]`，单位秒。

```python
from vadbench import create

vad = create("silero")          # webrtc | silero | funasr_fsmn | pyannote | firered
for seg in vad.detect("audio.wav"):
    print(seg.start, seg.end)
```

只装了部分引擎时，`available()` 只返回装好的那些，`unavailable()` 给出缺依赖的原因，不会因为某个可选依赖缺失而整体导入失败：

```python
import vadbench
vadbench.available()    # ['webrtc', ...]   已可用的引擎
vadbench.unavailable()  # {'pyannote': "No module named 'torch'"}
```

**刻意不做帧级概率 API。** pyannote 和 FireRed 内部有帧级分数，但 webrtc 和 FunASR 的封装并不统一暴露，硬凑出来就是「有的模型用真概率、有的用段边界反推的伪概率」，算出的 AUC 没有可比性。所以统一只比段级输出栅格化到 10 ms 帧之后的结果。

### 跑评测

```bash
# 先准备数据（各数据集下载/转换见 docs/eval.md）
python -m vadbench.eval.download_data --dataset alimeeting,alimeeting_eval

# 全量评测
python -m vadbench.eval --engines webrtc,silero,funasr_fsmn,firered,pyannote --dataset all

# 冒烟测试
python -m vadbench.eval --engines webrtc,silero --dataset aishell4 --max-files 1
```

输出 `results/eval_<stamp>.json`（含 per-file 混淆计数）与 `results/summary_<stamp>.csv`。

### 二次分析

```bash
python -m vadbench.report
```

重新读取 `results/eval_*.json`，输出每文件 F1 分布、FAR vs FP/N、跨数据集透视表，**不重新跑任何模型**。结果写到 `results/report/`。

同一 `(数据集, 引擎, 文件)` 有多次运行时取时间戳最新的那次，所以重跑单个引擎不需要手工维护「哪份结果才算数」。脚本还会把重算的 F1 与 `summary_*.csv` 对一遍，不一致就报警。

### 性能测试

```bash
python -m vadbench.perf --engines silero,firered,funasr_fsmn,pyannote,webrtc \
    --threads 1,2,4 --clip-s 10 --runs 50
```

输出 `results/perf_<stamp>.csv` / `.json`。`--skip-export` 跳过 ONNX 导出，`--skip-align` 跳过对齐校验。

### 自测

```bash
python tests/test_report.py           # 二次分析逻辑
python tests/check_minimal_install.py # 缺可选依赖时优雅降级
python tests/dump_tables.py           # 从 results/ 重新导出 Markdown 表格
```

`test_report.py` 覆盖聚合口径（时长加权 vs 文件平均）、重跑覆盖旧结果、离群文件不被均值掩盖、FAR 分母陷阱、旧版报告缺 `dataset` 字段的兼容。`dump_tables.py` 用来自查 README 里的数字是否仍与 `results/` 一致 —— 改过结果文件后值得跑一遍。

## 指标口径

预测段和参考段都按 10 ms 栅格化成二值帧序列（1 = 语音），逐帧对比：
```
P     = TP / (TP + FP)          # 预测为语音的帧中，真语音占比
R     = TP / (TP + FN)          # 参考语音帧中，被检出占比
F1    = 2*TP / (2*TP + FP + FN)
FAR   = FP / (FP + TN)          # 非语音帧上的虚警率（= FPR）
Miss  = FN / (TP + FN)          # 语音帧上的漏检率（= 1 − R）
```

两个容易写错的地方：

1. FAR 的分母是**全部非语音帧** `FP+TN`，不是 `TP+FP`。
2. `Miss = 1 − R`，所以同时报这两个指标是冗余的，放在一起只是为了方便看漏检。

汇总有两条路，会给出不同结论，**报数时必须写明用的是哪条**：

- **时长加权**（默认）：先把各文件的 TP/FP/TN/FN 分别求和再套公式，长文件权重大；
- **文件平均**：每文件先算 F1 再取算术平均，每场会议等权。CSV 里 `*_unweighted` 列是它。

## 仓库结构

```
vadbench/
  base.py            # BaseVad 抽象接口
  types.py           # SpeechSegment
  registry.py        # 引擎注册表 / create()
  audio.py           # 读取 + 重采样到 16 kHz mono
  download.py        # 权重下载 + 上游源码归档
  backends/          # 5 个引擎 + 4 个 ONNX 变体
  eval/              # 数据集加载、栅格化、指标、评测 runner
  perf/              # ONNX 导出、对齐校验、延迟压测
  report/            # results/eval_*.json 的二次分析
docs/
  eval.md            # 评测协议、数据准备
  vad_models.md      # 各引擎的官方链接、下载、许可证
  实验结果汇总.md      # 完整指标表
results/             # 原始评测结果（已提交，见下）
tests/               # 自测与表格导出
.github/workflows/   # CI：跑自测 + 校验 results/ 可复现
```

## 结果文件

`results/` 下的原始 JSON / CSV 已提交入库，仓库里的表格都由它们汇总而来：

| 文件 | 内容 |
|------|------|
| `eval_<stamp>.json` | 单次评测的 per-file 混淆计数 + 数据集汇总 |
| `summary_<stamp>.csv` | 数据集汇总（含 `*_unweighted` 文件平均列） |
| `summary_aishell4_merged.csv` | AISHELL-4 五引擎合并表 |
| `perf_<stamp>.csv` / `.json` | ONNX 延迟与对齐校验结果 |
| `report/report_<stamp>.json` | `vadbench.report` 的二次分析输出 |

早期几次 `eval_*.json` 的 `summary[].dataset` 字段是空的（那时还没加这个字段）。`vadbench.report` 会从同一份报告的 `per_file[].uri` 前缀反推归属，不需要回头改历史结果。

## 模型与数据

- `models/`、`vendor/`、`data/` 都不入库（见 `.gitignore`），用 `python -m vadbench.download` 和 `python -m vadbench.eval.download_data` 获取。
- 权重来源优先 ModelScope / 国内镜像，HF 直连失败时自动回退 `hf-mirror.com`。
- 数据集的许可证各不相同，请遵守各自条款。**MagicData-RAMC 为 CC-BY-NC-ND**，仅限非商业使用、禁止演绎。
- 各引擎的权重许可证见其托管页；本仓库代码为 MIT，见 [`LICENSE`](LICENSE)。

## 已知限制

- 只报段级指标，不做 AUC（原因见上文统一接口说明）。
- pyannote 的 `min_duration_on` / `min_duration_off` 都设为 0，即关掉了「过短语音段过滤」和「过短静音填充」两步后处理，纯 segmentation 原始输出。这几乎必然推高虚警 —— 它的高 FAR 里有相当一部分是配置选择，不是模型能力。
- RAMC 上所有引擎的 P 都偏低（0.89~0.95），其中包含 GT 口径差异：官方语音活动时间戳对呼吸声、笑声、短暂停顿的约定，大概率和会议集不一致，一部分「虚警」不是模型错。
- 所有引擎均未调参。真要选型应该在自己的数据上做一轮阈值 / 后处理扫描，而不是直接抄任何评测的排名 —— 包括这一篇。
- VoxConverse 因牛津源过慢未纳入（`vadbench/eval/datasets.py` 里保留了注册项）。

## 引用

本项目为个人评测记录。模型与数据版权归各自作者所有，引用请遵循各上游项目的许可。

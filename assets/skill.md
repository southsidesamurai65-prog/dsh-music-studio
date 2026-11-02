# 作曲：按参考风格写原创曲

把一个“像某某”的请求，变成**可测量的参数 + 可复现的合成脚本**。参考曲只提供风格特征，旋律与编曲必须原创。

## 何时使用

用户说“写一首像 X 的曲子”“想要 Y 风格”“这首太慢/太吵/不够江南，改一下”时。

## 工作流

0. **列编配清单**（在写简报之前）：按角色各选一件乐器——节奏组、主奏、低音、
   色彩、铺底、空间。清单必须能和最近做过的曲子逐项对比（见下节“抗同质化”）。
   没有节奏组的曲子（室内乐、纯民乐）也要显式写“无节奏组”。
1. **探测**：对每首参考曲调用工具 `music_style_probe`（路径参数为音频文件）。读回速度、调性、和弦循环、音色、空间、动态、结构。
   - 工具给出的是**假设**：和弦在无根音配置/重混响素材上会退化，若用户能提供和弦表或 MIDI，以用户为准。
   - 参考曲若不易获取，就用文字特征（BPM、调式、编制、年代）代替。
   - `arrangement` 字段是**关于“有没有节奏组”的线索**，不是事实：
     `onsets`（每秒起音数）高 = 编配密；`low-band pulse` 强且 `periodicity` 高 =
     低音有周期性脉冲（多半有底鼓/大鼓）；`backbeat` > 1.15 = 二四拍比一三拍重
     （流行鼓组的军鼓位）；`presence` 高 = 1–2 kHz 靠前（人声/主奏靠前）。
     这些数字**看不出**具体是钢琴还是古筝，必须配合频谱图和你自己的编配清单判断。
2. **写简报**：按六层展开（见下），每层给出具体数值或乐器名，并写明**不要什么**。把简报回给用户确认（若用户已明确，可直接开工）。
3. **合成**：新建 `compose_<name>.py`，`import synth_lib as sl`（路径见下），用 numpy 合成。结构上先定 BPM/调性/和弦，再写旋律，再加伴奏，最后混音。
   - 固定随机种子 `sl.seed(7)` 保证可复现。
   - 每件乐器写成 `(bar, beat, midi, dur, velocity)` 的数据表，方便改。
4. **验证**：对成品再跑一次 `music_style_probe`，对比 BPM、调性、频段平衡、宽度、动态弧与参考的差距；用 ffmpeg 画波形/频谱图检查结构与削波。
5. **迭代**：一次只改一层。用户说“太吵”改配器密度，“不够亮”改音区与高频，“不连贯”改句法（共同音衔接、动机贯穿），不要同时改五样。

## 六层简报模板

```text
【体裁/年代】lo-fi hip hop，2010s 末，卧室制作感
【参考】A《…》— 要它鼓的拖拽感；B《…》— 要它电钢琴的和弦色彩
【速度/律动】84 BPM，4/4，swing 约 55%，鼓点稀疏
【调性/和声】A 小调；Am9–Fmaj9–Cmaj9–G6/9，每和弦一小节
【编制】Rhodes、失谐锯齿铺底、无品贝斯、刷鼓
【音色/制作】暖，低通 ~8 kHz，磁带饱和，板式混响 ~2 s，宽度 0.3
【编配】节奏组：鼓组（底鼓 1/3、军鼓 2/4、闭镲八分）｜主奏：二胡｜低音：电贝斯
【结构】前奏 4 小节 → 主段 8 → 收尾 4；结尾留残响
【不要】失真吉他、808、人声、EDM riser
```

## 抗同质化：编配清单与差异检查

同一个题材做多了，最容易撞的不是旋律，而是**织体**：同样的“持续线条 + 拨弦琶音 + 长混响”。
开工前先写清单，收尾前逐项对比最近 2–3 首：

| 角色 | 本曲 | 最近一首 | 是否重复 |
|---|---|---|---|
| 节奏组（或“无”） | | | |
| 主奏 | | | |
| 低音 | | | |
| 色彩乐器 | | | |
| 铺底 | | | |
| 混音性格（干/湿、亮/暗） | | | |

**规则**：六项里如果有 4 项相同，就至少换一项，并且换的是**承重项**
（节奏组、主奏、低音），不是把古筝换成扬琴这种同一个角色内部的替换。

其它三条：

- 不复制上一首的脚本当起点。从 `templates/` 里挑一个**和上次不同的**模板新建，
  不要 `cp` 上一首的 `compose_*.py` 改参数。
- 每首曲子建自己的目录 `pieces/<name>/`，脚本、音频、图都放里面，方便日后 A/B。
- 混音参数每首重新核对（见“混音预设”），不要沿用上一首调好的那套。

## 合成库

`{{ASSETS}}/synth_lib.py`（`sl`）提供：

- 乐器：`guzheng` `dizi` `xiao` `erhu` `pipa_pluck` `pipa_tremolo` `yangqin` `sheng` `rhodes` `pad` `bass` `bass_pluck` `kick` `rim` `hat` `drum_low` `muyu` `bell` `gong`
- 空间：`make_ir` `convolve` `apply_reverb(mix, send, wet=…)`
- 母线：`add(bus, sig, t0, pan, gain)`；`master(mix, …)`；`write_wav(path, mix)`
- 工具：`midi2f` `fft_filter` `gate` `compress` `seed`

示例工程（可直接改）：`{{ASSETS}}/templates/lofi.py`、`{{ASSETS}}/templates/jiangnan.py`。
方法论文档：`{{ASSETS}}/STYLE_GUIDE.md`。

按角色挑乐器（不要再顺手拿上次那件）：

```python
import synth_lib as sl
print(sl.palette("pop"))     # 角色表 + 混音起点；还有 chamber / lofi / cinematic
```

- 主奏：`dizi` `xiao` `erhu` `violin` `cello` `piano` `rhodes` `choir`
- 节奏：`piano` `guzheng` `pipa_pluck` `yangqin` `rhodes`
- 低音：`bass` `bass_pluck` `ebass` `cello`
- 鼓组：`kick` `snare` `hat` `shaker` `clap` `rim` `drum_low` `muyu`
- 铺底：`pad` `strings` `choir` `sheng`
- 色彩：`guzheng` `pipa_pluck` `yangqin` `bell` `gong`
- 新加的非民乐件：`piano` `strings` `ebass` `snare` `shaker` `violin` `cello` `choir` `clap`

## 混音预设

`sl.MIX_PRESETS` 是**起点**（混响湿度/时长、压缩档、母带峰值、宽度增益、目标区间）：

| preset | 适用 | 混响 wet | 宽度目标 | crest 目标 |
|---|---|---|---|---|
| `chamber` | 民乐室内、丝竹 | 0.30 | 0.30–0.55 | 12–16 dB |
| `pop` | 流行、中国风流行 | 0.20 | 0.35–0.55 | 10–14 dB |
| `lofi` | lo-fi、卧室 | 0.25 | 0.15–0.35 | 12–18 dB |
| `cinematic` | 管弦、影视 | 0.42 | 0.35–0.60 | 14–20 dB |

用预设起步，然后**对成品跑探针**，把 `centroid_hz / width / crest_db` 和目标区间对齐；
对不上就改对应的一项，别把整套参数搬过来。

最小骨架：

```python
import numpy as np, synth_lib as sl
sl.seed(7)
BPM = 84; BEAT = 60 / BPM
mix = np.zeros((2, int(60 * 44100))); send = np.zeros_like(mix)
sl.add(mix, sl.guzheng(sl.midi2f(69), 1.0, 0.8), 0.0, -0.2, 1.0)
mix = sl.apply_reverb(mix, send, 0.4)
sl.write_wav("sketch.wav", sl.master(mix))
```

## 验收与交付

- 用 `ffmpeg -i out.wav -af ebur128 -f null -` 看响度（参考 −14 LUFS 左右）与峰值（≤ −1 dBFS）。
- 交付：mp3（320k）+ wav + 源码 + 波形图；说明每一版改了什么、为什么。
- 版本迭代时保留旧文件，便于 A/B。

## 边界

- 不复制参考曲的旋律、和声进行逐音或标志性音色；只学速度、调式、编制、织体、制作手法。
- 不要用艺人名字做商业宣传；商用前提示用户做相似性审查。
- 听不到声音时，用探针数字、频谱图与配器逻辑判断，并在交付说明里讲清哪些是推断。
- 不要把上一首的 `compose_*.py` 当模板复制；从 `templates/` 里挑一个不同的起步。

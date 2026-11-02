# 怎么让 AI 写出“像某几首歌”的音乐

> 结论先说：**“像 XX 那样”是最低效的说法。风格是一组可测量的参数，把它拆成 6 层、给参照 + 给约束，命中率会高一个数量级。**
> 如果你用的是我这种“从零合成”的方式，最可靠的路是：**参考音频 → 特征画像 → 参数化简报 → 逐层迭代**。本仓库的 [`style_probe.py`](../tools/style_probe.py) 就是把第一步自动化。

---

## 1. 先拆解：风格 = 6 层

学术界对“音乐风格”的定义本身就是分层的：体裁、演奏、混音可以分别建模、分别迁移（见文末 *Music Style Transfer: A Position Paper*）。实践中最有用的拆法：

| 层 | 具体参数 | 为什么重要 |
|---|---|---|
| 1. 体裁 / 年代 | 子流派、年代、地域 | 决定默认的“统计平均值” |
| 2. 速度 / 律动 | BPM、拍号、swing 比例、鼓的密度与位置 | 决定“身体感”，最难用形容词描述 |
| 3. 调性 / 和声 | 调式、和弦词汇（三和弦 / 七和弦 / 九和弦）、循环长度、转位 | 决定“色彩”，也是版权风险最高的层 |
| 4. 乐器 / 音色 | 编制、音区、失真度、滤波 | 决定“一听就认得出”的部分 |
| 5. 编排 / 结构 | 段落顺序、动态曲线、留白、drop | 决定“像一首完整的歌”还是“一段 loop” |
| 6. 制作 / 混音 | 混响空间、频响、饱和、立体声宽度、响度 | 最容易忽略，却最影响“像不像那个年代” |

**关键发现**：混音风格可以和作曲分离、单独从参考曲迁移（*Music mixing style transfer*, 2022）。所以“我想要这首歌的鼓声，但旋律完全不一样”是完全合理、也是法律上更安全的请求。

---

## 2. 对我（代码合成）最有效的 4 种给法

按“我一次就能做对”的概率从高到低：

### A. 参考曲 + 说清“喜欢哪一面”（性价比最高）
> “参考 X《…》的**鼓的摇摆**和**混响空间**；参考 Y《…》的**和弦色彩**；但速度要 90 BPM、要有女声。”

比“像 X 那样”强在很多倍，因为它把风格拆到了层。同一个参考曲可以只要第 2 层、不要第 3 层。

### B. 直接丢音频文件给我（我可以量化）
把 mp3/wav 放进工作目录，我运行：

```bash
python3 tools/style_probe.py reference.mp3 --json profile.json
```

输出是速度、调性、和弦候选、音色重心、频段平衡、立体声宽度、动态曲线、逐秒编排图——**这些能直接翻译成合成参数**。我的曲目实测：84 BPM、A 小调、warm/dark（重心 329 Hz）、宽度 0.24，与创作时设定一致。

注意：自动和弦识别在“无根音配置 + 密集旋律 + 重混响”的素材上会退化（我自己的 lo-fi 曲就是这种），输出里会同时给候选和弦并标注为估计值。要精确和声，用下一条。

### C. 给和弦 / MIDI / 简谱（最准）
一行 `Am9 | Fmaj9 | Cmaj9 | G6/9`，或一个 MIDI 文件，比任何描述都准。有 MIDI 时我可以逐音还原力度、时值、声部。

### D. 参数化简报（适合远程、异步、多轮）
见下一节模板。适合你先想清楚、我一次性实现，或者你用它去喂 Suno / Udio。

---

## 3. 风格简报模板（可直接复制）

```text
【体裁/年代】lo-fi hip hop，2010s 末，卧室制作感
【参考】A《…》— 要它鼓的拖拽感；B《…》— 要它电钢琴的和弦色彩
【速度/律动】84 BPM，4/4，swing 约 55%，鼓点稀疏、留白多
【调性/和声】A 小调；Am9–Fmaj9–Cmaj9–G6/9，每和弦一小节，4 小节循环
【编制】Rhodes 电钢琴、失谐锯齿铺底、无品贝斯、刷鼓、轻打击
【音色/制作】暖，低通到 ~8 kHz，磁带饱和 + 轻微 wow，板式混响 ~2 s，宽度 0.3
【结构】前奏 4 小节（无鼓）→ 主段 8 小节 → 收尾 4 小节淡出
【不要】失真吉他、808、人声、EDM riser、鼓的 drop
【时长/用途】45 秒，视频背景，无人声
```

每一项都对应可调参数，例如：`swing 55%` → 后半拍延迟 0.05 拍；`低通 8 kHz` → 一阶/二阶 FFT 滤波；`宽度 0.3` → 侧声道增益；`前奏 4 小节无鼓` → 编排表里删掉对应小节。

---

## 4. 三个提高命中率的习惯

1. **一次只改一个变量**：先定速度与和声，再定音色，最后调混音。同时改五样，你无法判断哪样起了作用。
2. **用参照系，不用形容词**：“温暖”对每个人都不同，“像 1970 年代踏板钢弦吉他、房间很干”则没有歧义。
3. **明确负向约束**：说“不要什么”往往比说“要什么”更有效。生成模型会填满所有留白，除非你明确禁止。

---

## 5. 如果你用 Suno / Udio / ElevenLabs 这类成品模型

它们的提示词逻辑和上面同构，但有平台特有的杠杆：

**五要素**（ElevenLabs 官方指南的总结）：任何提示词都在回答 *体裁、情绪、编制、速度、制作年代* 五个问题；你不回答，模型就选“最统计平均”的答案。例：

> `upbeat electronic track` → 平庸
> `French house, 122 BPM, filtered disco sample chops, sidechained bass, tight four-on-the-floor kick, rooftop-at-sunset mood, warm analog glue` → 有主见

**制作词汇会真实改变输出**：`sidechained`、`close-mic'd`、`bone-dry`、`tape saturation`、`plate reverb`、`wow and flutter`。不知道术语就描述空间：“像在楼梯间录的”。

**数字要写死**：BPM、调性（`in A minor`）、时长、`lyrics begin at 15 seconds`。官方明确说模型能相当精确地保持 BPM 和调性，方便你叠录。

**编排要“旁白”**：`start with just a shuffled drum loop, add a warm sub bassline after four bars, then bring in chopped vocal stabs` —— 关键词是 *start with / just / then / bring in*。做 loop 时用排除法：`four bars, no melody — just drums`。

**参考音频**：ElevenLabs 的 Audio Reference 支持约 30 秒上传，匹配 feel、groove、音色，不复制音符；上传会做版权检查，且官方定位是“你自己的音乐”。Udio 的 Styles 支持 1–2 个参考（歌曲或上传音频），有 Style Strength（默认 80%）和两首混合比例（默认 50/50），上传素材必须你有版权，且**基于上传素材生成的歌不能在 Udio 发布**。

**Udio 的提示词结构**：`主题 + 体裁标签 + 可选 "in the style of X"`；重复关键词可加权；歌词框里用 `[Verse]` `[Chorus]` `[Instrumental]` 等 guidance tags 控制段落；`Style Reduction` 用来削弱不想要的元素；`Manual Mode` 只用精确标签、不做自由发挥。

**Suno** 的具体功能与参数以其官方文档为准（本次未能访问核实）；上面这套五要素、制作词汇、结构标签的思路在同类模型上通用。

---

## 6. 版权与风险（别跳过）

- 学**风格、氛围、制作手法**通常没问题；**逐音复制旋律、照搬和声进行、克隆某位歌手的声音**风险高。
- 商用发行前做一次“相似性审查”：主旋律、hook、标志性音色是否可辨认。
- 用平台功能时注意规则：Udio 的“上传参考 → 不可发布”就是硬限制；ElevenLabs 的上传会过版权检查。
- 不要用艺人名字做商业宣传（“某某风格”可以，“某某本人”不行）。

---

## 7. 怎么验收“像”

1. **盲听 A/B**：把你的参考和成品随机顺序放给朋友听，问“哪首更贵/更旧/更深夜”。
2. **特征对齐**：用 `tools/style_probe.py` 分别测参考和成品，对比 BPM、调性、频段平衡、宽度、动态。数值接近不代表好听，但偏离往往就是“不像”的原因。
3. **结构检查**：段落顺序、留白、结尾处理是否一致——这是“loop”和“歌”的分界线。

---

## 8. 我这边的工作流

1. 你给参考曲（文件或链接描述）+ 一句“要哪一层”。
2. 我跑 `tools/style_probe.py` 得到画像，把不确定的部分（尤其和声）列成问题问你。
3. 我把画像翻译成合成参数，出第一版。
4. 你指“哪里不像”（越具体越好：鼓太紧 / 混响太大 / 和弦太爵士），我只改对应那层。
5. 通常 2–3 轮收敛。

---

## 参考来源

- ElevenLabs, *Music — Best practices*：<https://elevenlabs.io/docs/overview/capabilities/music/best-practices>
- Udio, *Prompt Like a Master*：<https://help.udio.com/en/articles/10716541-prompt-like-a-master>
- Udio, *Styles*：<https://help.udio.com/en/articles/11003046-styles>
- Udio, *Guidance tags*：<https://help.udio.com/en/articles/11166249-level-up-your-creations-with-guidance-tags>
- *Music Style Transfer: A Position Paper*, arXiv:1803.06841：<https://arxiv.org/abs/1803.06841>
- *Music Mixing Style Transfer* (contrastive, reference-based), arXiv:2211.02247：<https://arxiv.org/abs/2211.02247>
- *Stylus: training-free music style transfer* (reference audio), arXiv:2411.15913：<https://arxiv.org/abs/2411.15913>
- *Simple and Controllable Music Generation* (MusicGen, text + melody conditioning), arXiv:2306.05284：<https://arxiv.org/abs/2306.05284>
- *MusicGen-Chord* (chord conditioning), arXiv:2412.00325：<https://arxiv.org/abs/2412.00325>
- *Seed-Music* (fine-grained style control), arXiv:2409.09214：<https://arxiv.org/abs/2409.09214>

# VLA Episode Viewer (mcap)

Streamlit 页面：回放 VR 遥操录制的 mcap episode，对齐相机/state/action 数据，检测并可视化丢帧。

## 安装

```bash
pip install -r requirements.txt
```

## 目录结构

页面按如下模式扫描 episode：

```
<root>/<episode_id>/recording/*.mcap
```

例如：

```
data/
  20260706_111225_545157/
    recording/
      recording_0.mcap
```

如果 mcap 文件不在这个层级下，建一个软链接即可，不用移动原始文件：

```bash
mkdir -p data/episode1/recording
ln -s /path/to/your/recording_0.mcap data/episode1/recording/recording_0.mcap
```

## 运行

```bash
streamlit run app.py
```

默认数据目录是 `/ssd/data`，也可以用 `--data-path` 指定（注意 `--` 分隔符，把参数传给脚本本身而不是 streamlit）：

```bash
streamlit run app.py -- --data-path /your/mcap/root
```

打开侧边栏 "mcap root dir" 可以再改一次（默认值就是上面这个 root 路径），选择 episode 即可。

侧边栏 "Topic 配置" 里可以改相机/state/action 对应的 ROS topic 名字，不填的 topic 不会被读取。

## 页面说明

页面从上到下依次是：

- **Playback**：一个 Play/Pause 切换按钮 + FPS 拖动条 + Step 拖动条，再往下是 "Select state / action dims" 勾选框，然后是三个相机画面按 step 对齐显示（Play 期间只有这部分刷新，不会拖慢下面的图表）。
- **State / Action (Plotly)**：上面勾选的 state/action 维度曲线；丢帧处画竖线标注，每个 topic 的竖线各占独立的颜色和横向车道，避免不同 topic 的丢帧混在一起看不清。
- **Image Health**：三个相机各自的发布频率统计、丢帧次数/帧数、`coverage`（实际频率 / 相机标称 30Hz）。
- **Image Gap 明细**：每个相机具体丢在哪个 step 区间、丢了多少帧，按丢帧数从大到小排——想核实 Image Health 里的数字，把 Playback 的 Step 滑条拖到这里列出的区间就能直接看到对应的 "missing" 提示。
- **Joint Health**：state/action 各 topic 的同类统计，`coverage` 按关节/pose/target 标称 100Hz 折算。

## 丢帧判定逻辑（简述）

- 参考时钟固定为 `head` 相机；`head` 自己的时间戳会先用其它 topic 的时间戳补全开头/结尾，再对中段大间隔按它自己实际的中位数周期（不是固定标称 30Hz）插值补点，这样 head 自己的丢帧（包括开头结尾）也能被发现，且不会因为标称频率高于相机实际能力而系统性高估丢帧数。
- 其它每个 topic 独立对 head 的时间轴做最近邻查找，超出该 topic 自己的自适应容差就判定为 missing。
- `coverage` 列单独回答"整体是否达标"（用固定标称 30Hz/100Hz 折算），跟上面逐 step 的缺帧判定是两套独立的东西，不影响彼此（避免长期低速的 topic 被判成"到处都在丢帧"）。
- 一个 topic 的 `drop_frames` 总数可能几乎全部来自同一次连续的大 gap（比如一次 9 秒的中断，在 30Hz 下就是快 300 帧），不是分散在整段录制里的几百个独立瞬间——想看清具体是哪一段，查 **Image Gap 明细**（相机）或 Data Health 页以前的 gap 明细。

详见 `app.py` 顶部的 `NOMINAL_HZ_CAMERA`/`NOMINAL_HZ_JOINT`/`GAP_TOLERANCE_MULTIPLIER` 常量和 `_build_reference_timeline`/`_missing_runs` 的实现注释。

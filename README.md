# VLA Episode Viewer

Streamlit 可视化工具，支持两类 episode 数据：

- `mcap_vis.py`：回放 VR 遥操录制的 MCAP episode，对齐 ROS 相机/state/action 数据，检测并可视化丢帧。
- `lerobot_vis.py`：回放 LeRobot 数据集的 Parquet/MP4 episode，并展示图像、state/action、intervention 与数据健康统计。

## 安装

```bash
pip install -r requirements.txt
```

## MCAP Viewer

### 目录结构

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

### 运行

```bash
streamlit run mcap_vis.py
```

指定监听地址和端口（`0.0.0.0` 表示局域网内其他设备也能访问）：

```bash
streamlit run mcap_vis.py --server.address 0.0.0.0 --server.port 8501
```

默认数据目录是 `test`，也可以用 `--data-path` 指定（注意 `--` 分隔符，把参数传给脚本本身而不是 Streamlit）：

```bash
streamlit run mcap_vis.py -- --data-path /your/mcap/root
```

两者可以一起用：

```bash
streamlit run mcap_vis.py --server.address 0.0.0.0 --server.port 8501 -- --data-path /your/mcap/root
```

打开侧边栏 "mcap root dir" 可以再改一次（默认值就是上面这个 root 路径），选择 episode 即可。

侧边栏 "Topic 配置" 里可以改相机/state/action 对应的 ROS topic 名字，不填的 topic 不会被读取。

### 页面说明

页面从上到下依次是：

- **Playback**：一个 Play/Pause 切换按钮 + FPS 拖动条 + Step 拖动条，再往下是 "Select state / action dims" 勾选框，然后是三个相机画面按 step 对齐显示（Play 期间只有这部分刷新，不会拖慢下面的图表）。
- **State / Action (Plotly)**：上面勾选的 state/action 维度曲线；丢帧处画竖线标注，每个 topic 的竖线各占独立的颜色和横向车道，避免不同 topic 的丢帧混在一起看不清。
- **Image Health**：三个相机各自的发布频率统计、丢帧次数/帧数、`coverage`（实际频率 / 相机标称 30Hz）。
- **Image Gap 明细**：每个相机具体丢在哪个 step 区间、丢了多少帧，按丢帧数从大到小排——想核实 Image Health 里的数字，把 Playback 的 Step 滑条拖到这里列出的区间就能直接看到对应的 "missing" 提示。
- **Joint Health**：state/action 各 topic 的同类统计，`coverage` 按关节/pose/target 标称 100Hz 折算。

### 丢帧判定逻辑（简述）

- 参考时钟固定为 `head` 相机；`head` 自己的时间戳会先用其它 topic 的时间戳补全开头/结尾，再对中段大间隔按它自己实际的中位数周期（不是固定标称 30Hz）插值补点，这样 head 自己的丢帧（包括开头结尾）也能被发现，且不会因为标称频率高于相机实际能力而系统性高估丢帧数。
- 其它每个 topic 独立对 head 的时间轴做最近邻查找，超出该 topic 自己的自适应容差就判定为 missing。
- `coverage` 列单独回答"整体是否达标"（用固定标称 30Hz/100Hz 折算），跟上面逐 step 的缺帧判定是两套独立的东西，不影响彼此（避免长期低速的 topic 被判成"到处都在丢帧"）。
- 一个 topic 的 `drop_frames` 总数可能几乎全部来自同一次连续的大 gap（比如一次 9 秒的中断，在 30Hz 下就是快 300 帧），不是分散在整段录制里的几百个独立瞬间——想看清具体是哪一段，查 **Image Gap 明细**（相机）或 Data Health 页以前的 gap 明细。

详见 `mcap_vis.py` 顶部的 `NOMINAL_HZ_CAMERA`/`NOMINAL_HZ_JOINT`/`GAP_TOLERANCE_MULTIPLIER` 常量和 `_build_reference_timeline`/`_missing_runs` 的实现注释。

## LeRobot Viewer

### 目录结构

LeRobot viewer 使用 LeRobot 数据集的标准 Parquet/MP4 布局。数据根目录至少应包含元数据、episode 数据和相机视频：

```
<root>/
  meta/
    info.json
  data/
    chunk-000/
      file-000.parquet
  videos/
    observation.images.<camera_key>/
      chunk-000/
        file-000.mp4
```

`meta/info.json` 中的 `features` 用于自动发现 `observation.images.*` 视频相机，`video_path`（如存在）用于定位对应 MP4；未配置时使用 LeRobot 默认视频路径模板。

### 运行

默认数据目录是 `lerobot_data`：

```bash
streamlit run lerobot_vis.py
```

指定数据根目录：

```bash
streamlit run lerobot_vis.py -- --data-path /your/lerobot/dataset
```

指定监听地址、端口和数据根目录：

```bash
streamlit run lerobot_vis.py --server.address 0.0.0.0 --server.port 8501 -- --data-path /home/fiveages/data/lerobot_data/2026_07_10
```

打开侧边栏的 `dataset root dir` 可以在页面中修改数据根目录，再选择要回放的 episode。

### 页面说明

- **Img / State / Action**：Play/Pause、FPS 与 Step 控制；按 frame index 同步回放 `meta/info.json` 中发现的所有相机视频，并显示选中的 state/action 维度曲线。
- **Intervention**：数据包含 `observation.intervention` 时，当前 step 会显示 intervention 状态，曲线图中相应区间会高亮。
- **Image Health**：每个相机的视频帧可用性、频率、时间戳间隔、丢帧数及 coverage。
- **Signal Health**：`observation.state` 与 `action` 的同类统计；时间戳大间隔或相机视频缺帧会在信号中标为缺失。

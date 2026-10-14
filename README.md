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

同一个 episode 下的所有 MCAP 分片会合并读取，并按消息时间戳排序。

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

侧边栏 "Topic 配置" 明确指定数据来源：S 是机器人实际反馈（默认 `left/right/body_current_pose` 和 `/joint_states`），A 是 VR 控制指令（默认 `left/right_current_target` 和手控 topic）。只读取配置的 topic，不按名称自动分类。需要查看其他采集信号时，在对应输入框中追加 topic。

### 页面说明

页面从上到下依次是：

- **Playback**：Play/Pause、FPS、Step 和相机画面。播放期间刷新相机和窗口内曲线，健康统计无需逐帧重建。
- **State / Action (Plotly)**：全部已读取的数值维度均可选择，包括关节角 `position`、速度 `velocity`、力矩 `effort`、数组和布尔值；消息头中的时间戳等元数据不进入曲线选项。默认显示左右臂完整的 XYZ 对：`S:/left_current_pose.pose.position.x` 对应 `A:/left_current_target.pose.position.x`，右臂同理。对应字段相邻显示并使用同一色系，S 为实线，A 为浅色点线。两条曲线始终取各自 topic 的原始数据；没有对应指令的关节反馈仅显示 S。姿态、关节、手控等其他字段可手动选择。缺失值显示为断线，不填成 0；手控 topic 保留 session 外填 0 的规则。丢帧位置按 topic 分色标注。
- **Image Health**：配置相机各自的发布频率统计、丢帧次数/帧数、`coverage`（实际频率 / 相机标称 30Hz）。
- **Image Gap 明细**：每个相机具体丢在哪个 step 区间、丢了多少帧，按丢帧数从大到小排——想核实 Image Health 里的数字，把 Playback 的 Step 滑条拖到这里列出的区间就能直接看到对应的 "missing" 提示。
- **Signal Health**：state/action 各 topic 的同类统计。普通信号 `coverage` 统一按 100Hz 折算；手控 topic 按 session 活跃时间占比计算。

### 丢帧判定逻辑（简述）

- 参考时钟优先使用 `head`，没有 head 时使用其他有帧的相机；完全没有可用相机时使用消息最多的信号。参考时间戳会先用其它 topic 的时间戳补全开头/结尾，再对中段大间隔按自身实际的中位数周期（不是固定标称 30Hz）插值补点。
- 其它每个 topic 独立对参考时间轴做最近邻查找，超出该 topic 自己的自适应容差就判定为 missing。曲线是对齐后的采样视图，不代表逐条展示原始高频消息。
- `coverage` 列单独回答"整体是否达标"（用固定标称 30Hz/100Hz 折算），跟上面逐 step 的缺帧判定是两套独立的东西，不影响彼此（避免长期低速的 topic 被判成"到处都在丢帧"）。
- 一个 topic 的 `drop_frames` 总数可能几乎全部来自同一次连续的大 gap（比如一次 9 秒的中断，在 30Hz 下就是快 300 帧），不是分散在整段录制里的几百个独立瞬间——想看清具体是哪一段，查 **Image Gap 明细**（相机）或 Data Health 页以前的 gap 明细。

详见 `mcap_vis.py` 顶部的 `NOMINAL_HZ_CAMERA`/`NOMINAL_HZ_JOINT`/`GAP_TOLERANCE_MULTIPLIER` 常量和 `_build_reference_timeline`/`_missing_runs` 的实现注释。

MCAP 回归检查（使用合成录制文件测试真实解码与界面）：

```bash
python -m unittest discover -s tests -v
```

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

- **流畅播放**：浏览器连续播放并同步各相机视频；默认 1× 原速，可暂停、拖动 Step 或切换 0.5×/2×，选中的 state/action 曲线会显示同步移动的红线。
- **逐帧查看**：用 Step 滑条或左右方向键检查单帧画面及完整的 state/action 曲线。
- **Intervention**：数据包含 `observation.intervention` 时，当前 step 会显示 intervention 状态，曲线图中相应区间会高亮。
- **Image Health**：每个相机的视频帧可用性、频率、时间戳间隔、丢帧数及 coverage。
- **Signal Health**：`observation.state` 与 `action` 的同类统计；时间戳大间隔或相机视频缺帧会在信号中标为缺失。

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

侧边栏 "Topic 配置" 明确指定数据来源：S 读取 `/joint_states.position`，A 读取控制器统计消息中的 `command_interface.<joint>/position`，分别代表实测关节位置与控制链路中的关节位置指令。

兼容实际录制的两种统计格式：旧版 `/controller_manager/introspection_data/full` 中按 `Statistic.name` 提取指令；新版 `/controller_manager/introspection_data/values` 按 `names_version` 与 `/controller_manager/introspection_data/names` 配对。名称表会自动读取，即使名称表晚于第一条 values 到达也能正确匹配。不会把末端目标位姿当作关节 Action，也不会用 state 补出 Action。

### 页面说明

页面从上到下依次是：

- **流畅播放**：与 LeRobot 共用浏览器播放器，默认按时间戳原速，另可选择 30Hz / 60Hz（每秒播放 30 / 60 步），支持叠加 0.5×/2×、暂停、拖动 Step 和同步曲线红线。频率选择在当前浏览器会话内保留；例如 30Hz 数据以 60Hz 回放时约为原速两倍。首次打开 episode 生成临时 H.264 预览并复用缓存；需要带 libx264 的 `ffmpeg`，`install_env.sh` 会在缺少时安装。相机缺帧位置以黑底 missing 显示。
- **逐帧查看**：与 LeRobot 一样，使用 Step 或左右方向键检查单帧相机及完整 Plotly 曲线。
- **State / Action (Plotly)**：选项和图例按关节名显示，如 `S:body_joint3` / `A:body_joint3`。默认选前四对活动关节，S 为实线，A 为同色系浅色点线；与 LeRobot 使用相同的柔和配色，紫色改为浅紫。仅显示位置，排除速度、力矩、统计计数、限位标记、消息头、固定的 `body_joint1/2` 与 `head_joint1/2`，不添加模型补零维度。缺失数据以断线显示。无可用指令或名称表时明确提示。
- **Image Health**：配置相机各自的发布频率统计、丢帧次数/帧数、`coverage`（实际频率 / 相机标称 30Hz）。
- **Image Gap 明细**：每个相机具体丢在哪个 step 区间、丢了多少帧，按丢帧数从大到小排——想核实 Image Health 里的数字，把 Playback 的 Step 滑条拖到这里列出的区间就能直接看到对应的 "missing" 提示。
- **Signal Health**：反馈与指令各 topic 的同类统计，按这批录制的 500Hz 标称频率折算 `coverage`。

两种 viewer 共用关节选择保存逻辑：切换 episode 保留选择，也保留用户主动清空的状态。MCAP 按关节名保存选择，切换 `full` / `values` 指令格式时不重置。某段缺少的字段暂时隐藏，返回有该字段的 episode 后恢复。

图例中单击隐藏、双击单独查看的状态也按 S/A 关节名保存在当前浏览器会话中，切换 episode 或逐帧刷新时恢复；图表使用本机提供的 Plotly 脚本，不需要连接外部 CDN。

MCAP 支持只有关节数据、没有相机的 episode：按信号时间轴连续播放 S/A 曲线，支持暂停、拖动和 30Hz / 60Hz。纯关节播放不需要 ffmpeg；没有相机数据时显示说明，不将整段标记为图像丢帧。

2026-10-12 至 2026-10-14 的这批 XHand MCAP 各包含 44 个实际关节；排除上述四个固定关节后，界面是 40 对位置 S/A（躯干 2、双臂 14、双手 24）。这与旧 L6 LeRobot 数据的手部结构不同，不会通过删改手指或填零强凑成 28 个活动维度。

### 丢帧判定逻辑（简述）

- 参考时钟优先使用 `head`，没有 head 时使用其他有帧的相机；完全没有可用相机时使用消息最多的信号。参考时间戳会先用其它 topic 的时间戳补全开头/结尾，再对中段大间隔按自身实际的中位数周期（不是固定标称 30Hz）插值补点。
- 其它每个 topic 独立对参考时间轴做最近邻查找，超出该 topic 自己的自适应容差就判定为 missing。曲线是对齐后的采样视图，不代表逐条展示原始高频消息。
- `coverage` 列单独回答"整体是否达标"（用固定标称 30Hz/500Hz 折算），跟上面逐 step 的缺帧判定是两套独立的东西，不影响彼此（避免长期低速的 topic 被判成"到处都在丢帧"）。
- 控制器统计订阅可能首先带入录制前的旧缓存消息。这批文件有首条消息早 5–31 秒的情况；统计消息时间戳与录制时间相差超过 1 秒时会被排除并提示，避免将旧值显示为当前指令或错误拉长回放时间轴。
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

- **流畅播放**：浏览器连续播放并同步各相机视频；默认按时间戳原速，可选择 30Hz / 60Hz（每秒 30 / 60 步），叠加 0.5×/2×，也可暂停或拖动 Step。选中的 state/action 曲线使用与 MCAP 相同的柔和配色，并显示同步移动的红线。
- **逐帧查看**：用 Step 滑条或左右方向键检查单帧画面及完整的 state/action 曲线。
- **Intervention**：数据包含 `observation.intervention` 时，当前 step 会显示 intervention 状态，曲线图中相应区间会高亮。
- **Image Health**：每个相机的视频帧可用性、频率、时间戳间隔、丢帧数及 coverage。
- **Signal Health**：`observation.state` 与 `action` 的同类统计；时间戳大间隔或相机视频缺帧会在信号中标为缺失。

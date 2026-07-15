#!/usr/bin/env bash
set -euo pipefail

# Create the isolated environment used by run_viewer.sh.
ENV_NAME="${VIS_ENV_NAME:-lerobot_vis}"

if ! command -v conda >/dev/null 2>&1; then
    echo "错误：找不到 conda，请先安装 Miniforge/Miniconda 并加入 PATH。" >&2
    exit 1
fi

CONDA_BIN="$(command -v conda)"
export PYTHONNOUSERSITE=1

if ! "$CONDA_BIN" env list | awk -v name="$ENV_NAME" '$1 == name { found=1 } END { exit !found }'; then
    echo "创建 Conda 环境：$ENV_NAME (Python 3.12)"
    "$CONDA_BIN" create -n "$ENV_NAME" python=3.12 pip -y
else
    echo "Conda 环境已存在：$ENV_NAME"
fi

# Pin the application stack so a new machine gets the same versions as the
# working visualizer environment. Transitive dependencies are resolved by pip.
packages=(
    "streamlit==1.59.2"
    "numpy==2.2.6"
    "pyarrow==24.0.0"
    "pandas==3.0.3"
    "plotly==6.9.0"
    "matplotlib==3.11.0"
    "opencv-python-headless==4.11.0.86"
    "mcap==1.4.0"
    "mcap-ros2-support==0.5.7"
    "cffi==2.1.0"
)

echo "安装固定版本依赖..."
"$CONDA_BIN" run -n "$ENV_NAME" --no-capture-output \
    python -m pip install --upgrade --no-user --disable-pip-version-check "${packages[@]}"

echo "验证环境..."
"$CONDA_BIN" run -n "$ENV_NAME" --no-capture-output \
    python -c 'import sys, streamlit, google.protobuf, pyarrow, cv2, plotly, pandas, mcap, mcap_ros2.decoder; print("Python:", sys.executable); print("Streamlit:", streamlit.__version__); print("依赖验证通过")'

echo
echo "环境安装完成：$ENV_NAME"
echo "启动可视化："
echo "  ./run_viewer.sh"

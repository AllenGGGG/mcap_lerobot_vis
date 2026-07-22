#!/usr/bin/env bash
set -euo pipefail

# Resolve the real script path so this launcher also works through a symlink
# such as /home/fiveages/scripts/data_vis.sh.
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
ENV_NAME="${VIS_ENV_NAME:-lerobot_vis}"
DEFAULT_MCAP_PATH="/home/fiveages/data/mcap_data"
DEFAULT_LEROBOT_PATH="/home/fiveages/data/lerobot_data"
PORT="${VIS_PORT:-8501}"

if ! command -v conda >/dev/null 2>&1; then
    echo "错误：找不到 conda，请先运行 install_env.sh 或加载 Conda。" >&2
    exit 1
fi

CONDA_BIN="$(command -v conda)"
if ! "$CONDA_BIN" env list | awk -v name="$ENV_NAME" '$1 == name { found=1 } END { exit !found }'; then
    echo "错误：找不到 Conda 环境 $ENV_NAME，请先运行：./install_env.sh" >&2
    exit 1
fi

echo "请选择可视化程序："
echo "  1) MCAP viewer"
echo "  2) LeRobot viewer"
read -r -p "请输入 1 或 2 [2]: " choice
choice="${choice:-2}"

case "$choice" in
    1)
        app="mcap_vis.py"
        default_path="$DEFAULT_MCAP_PATH"
        echo "MCAP 默认路径：$DEFAULT_MCAP_PATH"
        ;;
    2)
        app="lerobot_vis.py"
        default_path="$DEFAULT_LEROBOT_PATH"
        echo "LeRobot 默认路径：$DEFAULT_LEROBOT_PATH"
        ;;
    *)
        echo "错误：只能输入 1 或 2。" >&2
        exit 2
        ;;
esac

read -r -p "请输入数据路径 [$default_path]: " data_path
data_path="${data_path:-$default_path}"

if [[ ! -d "$data_path" ]]; then
    echo "错误：数据路径不存在：$data_path" >&2
    exit 1
fi

if [[ "$choice" == "2" && ! -f "$data_path/meta/info.json" ]]; then
    shopt -s nullglob
    lerobot_meta_candidates=("$data_path"/*/meta/info.json)
    if [[ ${#lerobot_meta_candidates[@]} -eq 1 ]]; then
        data_path="${lerobot_meta_candidates[0]%/meta/info.json}"
        echo "检测到 LeRobot 数据集子目录，实际使用：$data_path"
    fi
fi

port_in_use() {
    ss -ltnH 2>/dev/null | awk -v port=":$1" '$4 ~ port "$" { found=1 } END { exit !found }'
}

original_port="$PORT"
while port_in_use "$PORT"; do
    PORT=$((PORT + 1))
done

if [[ "$PORT" != "$original_port" ]]; then
    echo "端口 $original_port 已被占用，改用端口 $PORT。"
fi

echo "正在启动 $app"
echo "数据路径：$data_path"
echo "访问地址：http://localhost:$PORT"

export PYTHONNOUSERSITE=1
exec "$CONDA_BIN" run -n "$ENV_NAME" --no-capture-output \
    python -m streamlit run "$SCRIPT_DIR/$app" \
    --server.address 0.0.0.0 \
    --server.port "$PORT" \
    --server.headless true \
    -- --data-path "$data_path"

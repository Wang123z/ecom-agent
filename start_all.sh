#!/usr/bin/env bash
# 一键启动全栈：MySQL + etcd + minio + milvus + vLLM + app
# 用法：bash start_all.sh
# 幂等：已在跑的服务自动跳过
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

MILVUS_DIR="${MILVUS_DIR:-/root/autodl-tmp/milvus}"
ADAPTER="${ADAPTER:-outputs/dpo_8b}"
MILVUS_PORT=19530
VLLM_PORT=6006
APP_PORT=8080

mkdir -p logs

if [ ! -d "$ADAPTER" ]; then
  echo "找不到 adapter: $ADAPTER"
  echo "可用：$(ls -d outputs/*/ 2>/dev/null | tr '\n' ' ')"
  exit 1
fi

wait_for() {
  local desc="$1" check="$2"
  for _ in $(seq 40); do
    eval "$check" > /dev/null 2>&1 && { echo "    $desc 就绪"; return 0; }
    sleep 3
  done
  echo "    $desc 超时，看 $PROJECT_DIR/logs/"
  return 1
}

if ! mysqladmin -uroot -proot ping > /dev/null 2>&1; then
  echo "==> 启动 MySQL"
  service mysql start
  wait_for "MySQL" "mysqladmin -uroot -proot ping"
fi

if ! redis-cli ping > /dev/null 2>&1; then
  echo "==> 启动 Redis"
  service redis-server start
  wait_for "Redis" "redis-cli ping"
fi

if ! pgrep -f "etcd --data-dir" > /dev/null; then
  echo "==> 启动 etcd"
  cd "$MILVUS_DIR"
  nohup ./etcd --data-dir ./etcd-data \
    --listen-client-urls http://0.0.0.0:2379 \
    --advertise-client-urls http://localhost:2379 > logs_etcd.log 2>&1 &
  wait_for "etcd" "./etcdctl --endpoints=localhost:2379 endpoint health"
fi

if ! pgrep -f "minio server" > /dev/null; then
  echo "==> 启动 minio"
  cd "$MILVUS_DIR"
  mkdir -p minio-data
  MINIO_ROOT_USER=minioadmin MINIO_ROOT_PASSWORD=minioadmin \
    nohup ./minio server ./minio-data --address :9000 --console-address :9001 > logs_minio.log 2>&1 &
  wait_for "minio" "curl -sf http://localhost:9000/minio/health/live"
fi

if ! pgrep -f "milvus run standalone" > /dev/null; then
  echo "==> 启动 milvus"
  export LD_LIBRARY_PATH="$MILVUS_DIR/milvus/lib:$LD_LIBRARY_PATH"
  cd "$MILVUS_DIR/milvus"
  nohup ./bin/milvus run standalone > "$MILVUS_DIR/logs_milvus.log" 2>&1 &
  wait_for "milvus" "python -c \"import socket;s=socket.socket();s.connect(('127.0.0.1',$MILVUS_PORT))\""
fi

cd "$PROJECT_DIR"

if ! curl -sf -m 2 "http://localhost:$VLLM_PORT/v1/models" > /dev/null; then
  echo "==> 启动 vLLM（约 2 分钟）"
  unset OMP_NUM_THREADS
  VLLM_USE_FLASHINFER_SAMPLER=0 PORT=$VLLM_PORT \
    nohup bash deploy/serve.sh "$ADAPTER" > logs/vllm.log 2>&1 &
  wait_for "vLLM" "curl -sf -m 2 http://localhost:$VLLM_PORT/v1/models"
fi

if ! curl -sf -m 2 "http://localhost:$APP_PORT/health" > /dev/null; then
  echo "==> 启动 app"
  nohup uvicorn app.main:app --host 0.0.0.0 --port $APP_PORT > logs/app.log 2>&1 &
  wait_for "app" "curl -sf -m 2 http://localhost:$APP_PORT/health"
fi

echo ""
echo "==> 全栈就绪"
echo "    Milvus   localhost:$MILVUS_PORT"
echo "    vLLM     http://localhost:$VLLM_PORT/v1"
echo "    App      http://localhost:$APP_PORT"
echo "    日志     $PROJECT_DIR/logs/"
echo ""
echo "试一下："
echo "  TOKEN=\$(curl -s -X POST http://localhost:$APP_PORT/api/login -H 'Content-Type: application/json' -d '{\"phone\":\"13800000000\",\"code\":\"1234\"}' | python -c \"import sys,json;print(json.load(sys.stdin)['token'])\")"
echo "  curl -N -X POST http://localhost:$APP_PORT/api/chat -H 'Content-Type: application/json' -H \"Authorization: Bearer \$TOKEN\" -d '{\"query\":\"退货要几天到账\",\"session_id\":\"s1\"}'"

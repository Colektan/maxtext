export DECOUPLE_GCLOUD=TRUE
export JAX_COORDINATOR_ADDRESS=localhost
export JAX_COORDINATOR_PORT=2222
export GPUS_PER_NODE=4
export NODE_RANK=0
export NNODES=1
python -m MaxText.train src/MaxText/configs/base.yml \
  run_name=test \
  base_output_directory=/mnt/caiyunuo/maxtext/temp_output \
  model_name=qwen3-0.6b-vl \
  dataset_type=synthetic \
  steps=10 \
  hardware=gpu \
  per_device_batch_size=3

python3 -m MaxText.train \
  src/MaxText/configs/base.yml \
  run_name=gpu01 \
  base_output_directory=/deps/output  \
  dataset_type=synthetic \
  enable_checkpointing=True \
  steps=10 \
  attention=cudnn_flash_te \
  scan_layers=False \
  use_iota_embed=True \
  hardware=gpu \
  per_device_batch_size=12
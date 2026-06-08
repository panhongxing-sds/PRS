# Run single greedy generation using TFB models
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# TokUR venv + large TMP on /HDDDATA (avoid filling /)
export PATH="/HDDDATA/phx/tokur_venv/bin:${PATH}"
export TMPDIR="${TMPDIR:-/HDDDATA/phx/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/HDDDATA/phx/pip-cache}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export TOKUR_MAX_MODEL_LEN="${TOKUR_MAX_MODEL_LEN:-4096}"
export TOKUR_MAX_NUM_BATCHED_TOKENS="${TOKUR_MAX_NUM_BATCHED_TOKENS:-4096}"

NUM_GPUS=8
GPU_IDS=(0 1 2 3 4 5 6 7)
# 单 seed 快跑：只看 96。若要论文 Table 多 seed 均值：改为 SEEDS=(96 89 64)
SEEDS=(96)

dataset="math500" # Options: math500, gsm8k_test, deepscaler, leg-counting
DATASET_START=0
DATASET_END=500
########################################################
# MATH500: 500 test samples
# GSM8K_TEST: 1300 test samples
# DEEPSCALER: 5000 test samples
# LEG_COUNTING: 100 test samples
########################################################

MODEL=qwen3b # Options: llama1b, llama8b, qwen3b

# =====================================================================
# Set MODEL_BASE_DIR to the directory containing your model checkpoints.
# You can download TFB models from HuggingFace:
#   - n1h111sm/TFB-Llama3.2-1B-Instruct
#   - n1h111sm/TFB-Meta-Llama-3.1-8B-Instruct
#   - n1h111sm/TFB-Qwen2.5-3B-Instruct
# =====================================================================
MODEL_BASE_DIR=${MODEL_BASE_DIR:-"./models"}

if [ "$MODEL" == "llama1b" ]; then
    model_path=TFB-Llama3.2-1B-Instruct
elif [ "$MODEL" == "llama8b" ]; then
    model_path=TFB-Meta-Llama-3.1-8B-Instruct
elif [ "$MODEL" == "qwen3b" ]; then
    model_path=TFB-Qwen2.5-3B-Instruct
else
    echo "Invalid model specified."
    exit 1
fi

echo "Using model path: $model_path"
echo "Using dataset: $dataset"

TOTAL=$((DATASET_END - DATASET_START))
CHUNK_SIZE=$((TOTAL / NUM_GPUS))

for SEED in ${SEEDS[@]}; do
for ((i=0; i<$NUM_GPUS; i++)); do
    GPU=${GPU_IDS[$i]}
    CHUNK_START=$((DATASET_START + i * CHUNK_SIZE))

    if [ $i -eq $((NUM_GPUS - 1)) ]; then
        CHUNK_END=$DATASET_END
    else
        CHUNK_END=$((CHUNK_START + CHUNK_SIZE))
    fi

    echo "Launching on GPU $GPU: start=$CHUNK_START, end=$CHUNK_END"

    CUDA_VISIBLE_DEVICES=$GPU HF_TOKEN=$HUGGING_FACE_HUB_TOKEN python run/greedy_unc_single_batch_refine.py \
        --dataset-path "datasets/$dataset.jsonl" \
        --dataset-start $CHUNK_START \
        --dataset-end $CHUNK_END \
        --model-path ${MODEL_BASE_DIR}/$model_path \
        --output-dir ./results/${MODEL}_results_vllm_pg/$dataset/seed$SEED/greedy_unc\
        --seed $SEED \
        --batch-size 16 \
        &
done
wait
done
#!/usr/bin/env bash
# Correctness validation: greedy (temp=0) decode on a low-entropy prompt, diffed
# byte-for-byte between the known-good reference (specprefill fork, untouched) and
# this new upstream-merged build. Two configs: plain, and full-stack (MTP + MoE
# cache + Speculative Prefill together).
set -uo pipefail

MODEL=/mnt/storage/models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
DRAFT=/mnt/storage/models/unsloth/Qwen3.8-Flash-Next-GGUF/MTP/Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf
SPF_MODEL=/mnt/storage/models/empero-ai/Qwen3.8-4B-Distill-GGUF/Qwen3.8-4B-Q4_K_M.gguf
REF_BIN=/home/mark/scripts/llama.cpp-qwen4exp-rdnaport-specprefill/build-stage2/bin/llama-server
NEW_BIN=/home/mark/scripts/llama.cpp-qwen4exp-rdnaport-latest/build-stage2/bin/llama-server
PORT=8960
OUT=/home/mark/scripts/llama.cpp-qwen4exp-rdnaport-latest/correctness-results.txt

export ROCR_VISIBLE_DEVICES=0,1
export HIP_VISIBLE_DEVICES=0,1
export GGML_OP_OFFLOAD_MIN_BATCH=128
export LLAMA_ATTN_ROT_DISABLE=1

> "$OUT"

PROMPT_PLAIN="List the first 20 prime numbers, one per line, nothing else."
PROMPT_NEEDLE="The secret passcode is ZQX-7734-MOON. Remember this. $(python3 -c 'print("The quick brown fox jumps over the lazy dog. " * 400)') What is the secret passcode mentioned at the start of this text?"

run_config() {
    local label="$1"; local bin="$2"; shift 2
    local log="/home/mark/scripts/llama.cpp-qwen4exp-rdnaport-latest/corr-${label}.log"
    echo "=== starting $label ($bin) ===" | tee -a "$OUT"
    "$bin" --port $PORT --model "$MODEL" "$@" \
        --host 127.0.0.1 --ctx-size 32768 --n-gpu-layers 999 --split-mode layer \
        --tensor-split 87,13 --main-gpu 0 --flash-attn on --threads 6 --parallel 1 \
        --n-cpu-moe 42 --override-tensor "per_layer_token_embd.*=CPU" \
        --jinja --timeout 300 --log-disable > "$log" 2>&1 &
    local pid=$!
    for i in $(seq 1 180); do
        if curl -s -m 2 "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q "ok"; then break; fi
        if ! kill -0 $pid 2>/dev/null; then
            echo "$label: died during startup" | tee -a "$OUT"; tail -40 "$log" | tee -a "$OUT"; return 1
        fi
        sleep 1
    done
    sleep 2

    local plain_out
    plain_out=$(curl -s -m 120 "http://127.0.0.1:${PORT}/completion" \
        -H "Content-Type: application/json" \
        -d "$(python3 -c "import json,sys; print(json.dumps({'prompt': sys.argv[1], 'n_predict': 200, 'temperature': 0, 'top_k': 1}))" "$PROMPT_PLAIN")" \
        | python3 -c "import json,sys; print(json.load(sys.stdin).get('content',''))")
    echo "$plain_out" > "/home/mark/scripts/llama.cpp-qwen4exp-rdnaport-latest/corr-${label}-plain.txt"
    echo "$label plain output saved (${#plain_out} chars)" | tee -a "$OUT"

    local needle_out
    needle_out=$(curl -s -m 180 "http://127.0.0.1:${PORT}/completion" \
        -H "Content-Type: application/json" \
        -d "$(python3 -c "import json,sys; print(json.dumps({'prompt': sys.argv[1], 'n_predict': 60, 'temperature': 0, 'top_k': 1}))" "$PROMPT_NEEDLE")" \
        | python3 -c "import json,sys; print(json.load(sys.stdin).get('content',''))")
    echo "$needle_out" > "/home/mark/scripts/llama.cpp-qwen4exp-rdnaport-latest/corr-${label}-needle.txt"
    echo "$label needle output: $needle_out" | tee -a "$OUT"

    kill -9 $pid 2>/dev/null
    pkill -9 -f "port $PORT" 2>/dev/null
    sleep 3
}

echo "--- PLAIN CONFIG (no spec-decode, no cache, no spec-prefill) ---" | tee -a "$OUT"
run_config "ref-plain" "$REF_BIN"
run_config "new-plain" "$NEW_BIN"

echo "--- FULL-STACK CONFIG (MTP + MoE cache + Speculative Prefill) ---" | tee -a "$OUT"
FULL_ARGS=(-md "$DRAFT" --spec-type ngram-mod,draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 --spec-draft-ngl 999 \
    --moe-expert-cache-experts 110 --moe-expert-cache-inserts 4 \
    --spec-prefill --spec-prefill-draft-model "$SPF_MODEL" --spec-prefill-p 0.30 --spec-prefill-draft-ngl 999 \
    --spec-prefill-draft-device ROCm0 --spec-prefill-draft-ctx 4096)
run_config "ref-full" "$REF_BIN" "${FULL_ARGS[@]}"
run_config "new-full" "$NEW_BIN" "${FULL_ARGS[@]}"

echo "=== DIFFS ===" | tee -a "$OUT"
for cfg in plain full; do
    echo "--- $cfg (plain-prompt) diff ---" | tee -a "$OUT"
    diff "corr-ref-${cfg}-plain.txt" "corr-new-${cfg}-plain.txt" > /dev/null 2>&1 && echo "IDENTICAL" | tee -a "$OUT" || { echo "DIFFERS:" | tee -a "$OUT"; diff "corr-ref-${cfg}-plain.txt" "corr-new-${cfg}-plain.txt" | tee -a "$OUT"; }
    echo "--- $cfg (needle) content ---" | tee -a "$OUT"
    echo "ref:  $(cat corr-ref-${cfg}-needle.txt)" | tee -a "$OUT"
    echo "new:  $(cat corr-new-${cfg}-needle.txt)" | tee -a "$OUT"
done

echo "CORRECTNESS_VALIDATION_DONE" | tee -a "$OUT"

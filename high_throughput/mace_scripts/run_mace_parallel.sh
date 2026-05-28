#!/bin/bash
# ============================================================
# run_mace_parallel.sh — Parallel MACE relaxation across GPUs
#
# Usage:
#   # Slab relaxation (Step 2)
#   bash run_mace_parallel.sh slab "./2_slabs/*/*/surf_*"
#
#   # Adsorption relaxation (Step 4)
#   bash run_mace_parallel.sh ads "./3_adsorption/*/ads_*"
#
#   # Custom GPU selection
#   GPUS="0,3" bash run_mace_parallel.sh slab "./2_slabs/*/*/surf_*"
# ============================================================

MODE="${1:-slab}"           # slab or ads
PATTERN="${2:-./2_slabs/*/*/surf_*}"

# ===== Configuration =====
GPUS="${GPUS:-0,1,2,3}"    # Available GPU IDs (override with env var)
MODEL_PATH="${MODEL_PATH:-/home/qiugt/software/mace/mace-mpa-0-medium.model}"
FMAX="${FMAX:-0.05}"
MAX_STEPS="${MAX_STEPS:-300}"
OPTIMIZER="${OPTIMIZER:-FIRE}"
DTYPE="${DTYPE:-float64}"
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"  # Processes per GPU (1 is safest)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MACE_RELAX="${SCRIPT_DIR}/mace_relax.py"

# ===== Parse GPU list =====
IFS=',' read -ra GPU_LIST <<< "${GPUS}"
NUM_GPUS=${#GPU_LIST[@]}
TOTAL_PROCS=$((NUM_GPUS * PROCS_PER_GPU))

echo "============================================================"
echo " MACE-MP-0 Parallel Relaxation"
echo "============================================================"
echo "  Mode:       ${MODE}"
echo "  Pattern:    ${PATTERN}"
echo "  GPUs:       ${GPUS} (${NUM_GPUS} GPUs × ${PROCS_PER_GPU} proc = ${TOTAL_PROCS} workers)"
echo "  Model:      ${MODEL_PATH}"
echo "  fmax:       ${FMAX}, max_steps: ${MAX_STEPS}, optimizer: ${OPTIMIZER}"
echo ""

# ===== Find all directories =====
DIRS=($(ls -d ${PATTERN} 2>/dev/null | sort))
TOTAL=${#DIRS[@]}

if [ "${TOTAL}" -eq 0 ]; then
    echo "No directories found matching: ${PATTERN}"
    exit 1
fi

echo "  Found ${TOTAL} directories to process"

# ===== Split into chunks =====
CHUNK_SIZE=$(( (TOTAL + TOTAL_PROCS - 1) / TOTAL_PROCS ))
TMPDIR=$(mktemp -d)

for ((i=0; i<TOTAL_PROCS; i++)); do
    START=$((i * CHUNK_SIZE))
    END=$((START + CHUNK_SIZE))
    if [ ${END} -gt ${TOTAL} ]; then END=${TOTAL}; fi

    CHUNK_FILE="${TMPDIR}/chunk_${i}.txt"
    for ((j=START; j<END; j++)); do
        echo "${DIRS[$j]}" >> "${CHUNK_FILE}"
    done
done

echo "  Split into ${TOTAL_PROCS} chunks of ~${CHUNK_SIZE} each"
echo ""

# ===== Launch processes =====
mkdir -p logs

PIDS=()
for ((i=0; i<TOTAL_PROCS; i++)); do
    CHUNK_FILE="${TMPDIR}/chunk_${i}.txt"
    if [ ! -f "${CHUNK_FILE}" ]; then continue; fi

    GPU_ID=${GPU_LIST[$((i % NUM_GPUS))]}
    LOGFILE="logs/mace_${MODE}_gpu${GPU_ID}_proc${i}.log"

    echo "  Worker ${i}: GPU ${GPU_ID}, $(wc -l < ${CHUNK_FILE}) dirs → ${LOGFILE}"

    # Build directory list as pattern-like args
    DIRS_FOR_PROC=$(cat "${CHUNK_FILE}" | tr '\n' ' ')

    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 "${MACE_RELAX}" \
        --batch --pattern "THIS_IS_IGNORED" \
        --mode "${MODE}" \
        --model_path "${MODEL_PATH}" \
        --fmax "${FMAX}" \
        --max_steps "${MAX_STEPS}" \
        --optimizer "${OPTIMIZER}" \
        --dtype "${DTYPE}" \
        --skip_converged \
        $(for d in ${DIRS_FOR_PROC}; do echo "--calc_dir $d"; done | head -1) \
        > "${LOGFILE}" 2>&1 &

    # Actually, the batch pattern approach is simpler. Let's use a wrapper.
    # Kill the above and use a per-directory loop instead.
    kill $! 2>/dev/null

    # Launch a simple loop script
    (
        export CUDA_VISIBLE_DEVICES=${GPU_ID}
        while IFS= read -r dir; do
            [ -z "$dir" ] && continue
            python3 "${MACE_RELAX}" \
                --calc_dir "$dir" \
                --mode "${MODE}" \
                --model_path "${MODEL_PATH}" \
                --fmax "${FMAX}" \
                --max_steps "${MAX_STEPS}" \
                --optimizer "${OPTIMIZER}" \
                --dtype "${DTYPE}" \
                --skip_converged
        done < "${CHUNK_FILE}"
    ) > "${LOGFILE}" 2>&1 &

    PIDS+=($!)

    # Stagger GPU starts to avoid model loading race
    if [ $i -lt $((TOTAL_PROCS - 1)) ]; then
        NEXT_GPU=${GPU_LIST[$(( (i+1) % NUM_GPUS ))]}
        if [ "${NEXT_GPU}" == "${GPU_ID}" ]; then
            sleep 5  # Same GPU, small delay
        else
            sleep 2  # Different GPU
        fi
    fi
done

echo ""
echo "============================================================"
echo "  All ${#PIDS[@]} workers launched"
echo ""
echo "  Monitor:  tail -f logs/mace_${MODE}_*.log"
echo "  Progress: grep -c 'OK\|FAIL' logs/mace_${MODE}_*.log"
echo "  Kill all: kill ${PIDS[*]}"
echo "============================================================"

# Wait for all
wait
echo ""
echo "All workers finished!"
rm -rf "${TMPDIR}"

# Count results
DONE=$(find ${PATTERN} -name "energy.json" 2>/dev/null | wc -l)
echo "Completed: ${DONE}/${TOTAL} directories have energy.json"

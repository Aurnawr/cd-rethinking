#!/usr/bin/env bash
# Live THRONE progress dashboard — reads tqdm output from run.log for real-time count.
OUT_ROOT="${OUT_ROOT:-/teamspace/studios/this_studio/cd-rethinking/outputs/throne}"
LOG="${OUT_ROOT}/run.log"
TOTAL=5000

while true; do
    clear
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    printf "║  THRONE EXPERIMENT — %-47s ║\n" "$(date '+%Y-%m-%d %H:%M:%S UTC')"
    echo "╠══════════════════════════════════════════════════════════════════════╣"
    printf "║  %-6s  %-12s  %-8s  %-7s  %-12s  %-8s ║\n" "Method" "Done/Total" "Left" "%" "Speed" "Status"
    echo "╠══════════════════════════════════════════════════════════════════════╣"

    CURRENT_METHOD=""
    CURRENT_DONE=0
    CURRENT_SPEED=""
    # Parse last tqdm line from log to get live count for the running method
    if [ -f "${LOG}" ]; then
        # tqdm writes e.g. "| 29/5000 [02:12<6:47:11,  4.91s/it]"
        LAST_TQDM=$(grep -oP '\|\s+\d+/5000[^|]+s/it' "${LOG}" 2>/dev/null | tail -1)
        if [ -n "${LAST_TQDM}" ]; then
            CURRENT_DONE=$(echo "${LAST_TQDM}" | grep -oP '\d+(?=/5000)')
            CURRENT_SPEED=$(echo "${LAST_TQDM}" | grep -oP '[\d.]+s/it' | tail -1)
        fi
        # Find which method is currently running
        CURRENT_METHOD=$(grep ">>> Step 1" "${LOG}" | tail -1 | grep -oP '\[.*?\]' | tr -d '[]')
    fi

    for METHOD in none vcd icd sid apc; do
        DIR="${OUT_ROOT}/${METHOD}"
        FINAL="${DIR}/responses.json"
        CKPT="${DIR}/responses.ckpt.json"

        if [ -f "${FINAL}" ]; then
            DONE=$(python3 -c "import json; d=json.load(open('${FINAL}')); print(len(d['responses']))" 2>/dev/null || echo "${TOTAL}")
            STATUS="✓ DONE"
            SPEED=""
        elif [ "${METHOD}" = "${CURRENT_METHOD}" ]; then
            DONE="${CURRENT_DONE}"
            STATUS="► running"
            SPEED="${CURRENT_SPEED}"
        elif [ -f "${CKPT}" ]; then
            DONE=$(python3 -c "import json; d=json.load(open('${CKPT}')); print(len(d['responses']))" 2>/dev/null || echo "0")
            STATUS="► running"
            SPEED="${CURRENT_SPEED}"
        else
            DONE=0
            STATUS="  waiting"
            SPEED=""
        fi

        if [ "${DONE}" -gt 0 ] 2>/dev/null; then
            PCT=$(python3 -c "print(f'{${DONE}/${TOTAL}*100:.1f}')" 2>/dev/null || echo "0.0")
            REM=$((TOTAL - DONE))
        else
            PCT="0.0"; REM=${TOTAL}; DONE=0
        fi

        printf "║  %-6s  %5d/%-5d   %5d    %6s%%  %-12s  %-8s ║\n" \
            "${METHOD}" "${DONE}" "${TOTAL}" "${REM}" "${PCT}" "${SPEED}" "${STATUS}"
    done

    echo "╠══════════════════════════════════════════════════════════════════════╣"
    # ETA for current method
    if [ -n "${CURRENT_SPEED}" ] && [ "${CURRENT_DONE}" -gt 0 ] 2>/dev/null; then
        REM_CUR=$((TOTAL - CURRENT_DONE))
        # extract numeric seconds from e.g. "3.95s/it"
        SPD_NUM=$(echo "${CURRENT_SPEED}" | grep -oP '[\d.]+')
        ETA_S=$(python3 -c "print(int(${REM_CUR} * ${SPD_NUM}))" 2>/dev/null || echo "?")
        ETA_H=$(python3 -c "s=${ETA_S}; print(f'{s//3600}h {(s%3600)//60}m')" 2>/dev/null || echo "?")
        printf "║  Current method [%-4s]: %5d left  ETA ≈ %-27s ║\n" \
            "${CURRENT_METHOD}" "${REM_CUR}" "${ETA_H}"
    fi
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo "  Window 0: generation  |  Ctrl+C exits dashboard (run continues)"
    sleep 10
done

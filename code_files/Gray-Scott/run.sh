LOG_DIR="logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

run_script() {
    local script="$1"
    local name="${script%.py}"
    local log_file="$LOG_DIR/train.log"

    echo "========================================"
    echo "Starting $script ... Logging to $log_file"
    echo "========================================"

    echo "Script   : $script" | tee "$log_file"
    echo "Started  : $(date)"  | tee -a "$log_file"
    echo "----------------------------------------" | tee -a "$log_file"

    python -u "$script" 2>&1 | tee -a "$log_file"

    local exit_code=${PIPESTATUS[0]}

    echo "----------------------------------------" | tee -a "$log_file"
    echo "Finished : $(date)"   | tee -a "$log_file"
    echo "Exit code: $exit_code" | tee -a "$log_file"

    if [ $exit_code -eq 0 ]; then
        echo "[$script] COMPLETED successfully -> $log_file"
    else
        echo "[$script] FAILED (exit $exit_code) -> $log_file"
    fi

    echo ""
}

run_script "lino_gs.py"


echo "All scripts finished. Logs saved in: $LOG_DIR/"
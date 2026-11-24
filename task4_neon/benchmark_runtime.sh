
#!/bin/bash
set -euo pipefail

PROMPT="Once upon a time, there was a little robot named Alex."

mkdir -p results

echo "Round,Runtime,Tokens_per_sec" > results/runtime_benchmark.csv

for i in {1..5}; do
    echo "===== Round $i ====="

    # Alternate order to reduce systematic measurement bias
    if (( i % 2 == 1 )); then
        order=("scalar" "neon")
    else
        order=("neon" "scalar")
    fi

    for mode in "${order[@]}"; do
        ./run_"$mode" stories15M.bin \
            -i "$PROMPT" -t 0 -n 128 -s 42 \
            > /dev/null \
            2> results/"${mode}_${i}.txt"

        speed=$(awk '/achieved tok\/s:/ {print $3}' \
            results/"${mode}_${i}.txt")

        echo "$i,$mode,$speed" >> results/runtime_benchmark.csv
        echo "$mode: $speed tokens/sec"
    done
done

echo "===== Benchmark Completed ====="
cat results/runtime_benchmark.csv

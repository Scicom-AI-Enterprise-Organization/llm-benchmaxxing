#!/bin/bash

yq eval '.runs[]' config.yaml -o=json | jq -c '.' | while read run; do
    NAME=$(echo $run | jq -r '.name')
    MODEL_PATH=$(echo $run | jq -r '.model_path')
    PORT=$(echo $run | jq -r '.port')
    OUTPUT_DIR=$(echo $run | jq -r '.output_dir')
    CTX_VALUES=($(echo $run | jq -r '.context_size[]'))
    CONCURRENCY_VALUES=($(echo $run | jq -r '.concurrency[]'))
    NUM_PROMPTS_VALUES=($(echo $run | jq -r '.num_prompts[]'))
    OUTPUT_LEN_VALUES=($(echo $run | jq -r '.output_len[]'))
    
    # Loop through tp_dp_pairs
    echo "$run" | jq -c '.tp_dp_pairs[]' | while read pair; do
        TP=$(echo $pair | jq -r '.tp')
        DP=$(echo $pair | jq -r '.dp')
        PP=$(echo $pair | jq -r '.pp')
        
        for CTX in "${CTX_VALUES[@]}"; do
            for CONCURRENCY in "${CONCURRENCY_VALUES[@]}"; do
                for NUM_PROMPTS in "${NUM_PROMPTS_VALUES[@]}"; do
                    for OUTPUT_LEN in "${OUTPUT_LEN_VALUES[@]}"; do
                        RESULT_NAME="${NAME}_TP${TP}_DP${DP}_CTX${CTX}_C${CONCURRENCY}_P${NUM_PROMPTS}_O${OUTPUT_LEN}"
                        
                        echo ""
                        echo "================================================================"
                        echo "RUNNING BENCHMARK: $NAME"
                        echo "TP: $TP | DP: $DP | PP: $PP | CONCURRENCY: $CONCURRENCY | CONTEXT SIZE: $CTX"
                        echo "NUM_PROMPTS: $NUM_PROMPTS | OUTPUT_LEN: $OUTPUT_LEN"
                        echo "================================================================"
                        
                        vllm bench serve \
                            --backend vllm \
                            --base-url http://localhost:$PORT \
                            --model $MODEL_PATH \
                            --endpoint /v1/completions \
                            --dataset-name random \
                            --random-input-len $CTX \
                            --random-output-len $OUTPUT_LEN \
                            --num-prompts $NUM_PROMPTS \
                            --max-concurrency $CONCURRENCY \
                            --request-rate inf \
                            --ignore-eos \
                            --save-result \
                            --result-dir $OUTPUT_DIR \
                            --result-filename "${RESULT_NAME}.json" \
                            --percentile-metrics ttft,tpot,itl,e2el
                    done
                done
            done
        done
    done
done
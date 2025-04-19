#!/usr/bin/env bash
# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error
# context="nocontext forgetdifference focusdifference"

# Run different models and settings
# num_type='singleDIY'
num_type='single'

context="noContext"
models="llama3.1-8b"
# models="mistral-7B Qwen2.5-7B llama3.1-8b llama2-7b gemma2-9b Phi3.5-4b"

# Define the output directory
token_method="last"
output_dir="outputs_${token_method}"
results_dir="results"
# results_dir="results"

# Run the evaluation, if run probing, add 
# --is_probing 
# to the command below

for c in $context; do
    for m in $models; do
        echo `date`, Evaluating $m with context: $c ...
        python run_eval.py \
            --model $m \
            --context $c \
            --num_type $num_type \
            --output_dir $output_dir \
            --token_method $token_method \
            --results_dir $results_dir \
            # --is_probing
    done
done
  
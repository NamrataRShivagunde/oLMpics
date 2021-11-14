#!/bin/bash

# TRAIN_DATA=('antonym_synonym_negation_train.jsonl' 'coffee_cats_quantifiers_train.jsonl' 'compositional_comparison_train.jsonl' 'hypernym_conjunction_train.jsonl' 'number_comparison_age_compare_masked_train.jsonl' 'size_comparison_train.jsonl')
# EVAL_DATA=('antonym_synonym_negation_dev.jsonl' 'coffee_cats_quantifiers_dev.jsonl' 'compositional_comparison_dev.jsonl' 'hypernym_conjunction_dev.jsonl' 'number_comparison_age_compare_masked_dev.jsonl' 'size_comparison_dev.jsonl')
# NUM_CHOICES=(2 5 3 3 2 2)

TRAIN_DATA=('antonym_synonym_negation_train.jsonl')
EVAL_DATA=('antonym_synonym_negation_dev.jsonl')
NUM_CHOICES=(2)

MODELS=('gpt2')

for seed in 123; do
    for model in "${MODELS[@]}"; do
        for i in "${!TRAIN_DATA[@]}"; do
            python finetune_gpt2_mlm.py $model "data/${TRAIN_DATA[i]}" "data/${EVAL_DATA[i]}" "${NUM_CHOICES[i]}" \
            --learning_rate 5e-5 \
            --num_train_epochs 2
        done
    done
done


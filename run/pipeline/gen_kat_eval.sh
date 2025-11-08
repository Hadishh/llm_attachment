#!/bin/sh

EXPERIMENT_DIR=$1
GEN_MODEL_NAME=$2
DATASET_NAME=$3
GENERATION_PROMPT=$4
GENERATION_KNOWLEDGE_TYPE=${5:-"triples"}
BATCH_SIZE=${6:-128}
MAX_NEW_TOKENS=${7:-16384}


GENERATION_OTUPUT=${EXPERIMENT_DIR}/${GEN_MODEL_NAME}_generations.json
echo "GENERATING ..."
echo "PLanning to save to: ${GENERATION_OTUPUT}"


python -m src.main --seed 42 \
    --model ${GEN_MODEL_NAME} \
    --dataset ${DATASET_NAME} \
    --num_gpus 4 \
    --output $GENERATION_OTUPUT \
    --prompt ${GENERATION_PROMPT} \
    --knowledge_type ${GENERATION_KNOWLEDGE_TYPE} \
    --max_new_tokens ${MAX_NEW_TOKENS} --batch_size ${BATCH_SIZE}

## build KAT dataset for evaluation
mkdir -p ${EXPERIMENT_DIR}/kat_dataset
KAT_DATASET_OUTPUT=${EXPERIMENT_DIR}/kat_dataset/${GEN_MODEL_NAME}_kat.jsonl

python -m src.build_kat_dataset_from_generations \
    --generations $GENERATION_OTUPUT --output $KAT_DATASET_OUTPUT

## run KAT evaluation
KAT_CHUNKS_FOLDER=${EXPERIMENT_DIR}/kat_llm_eval_chunks
mkdir -p ${KAT_CHUNKS_FOLDER}

echo "EVALUATING KAT ..."

python -m src.run_kat \
    --jsonl_path $KAT_DATASET_OUTPUT \
    --prompt prompts/KAT_prompt.txt \
    --output_dir $KAT_CHUNKS_FOLDER \
    --num_gpus 4 --model deepseek-r1-32b \
    --max_new_tokens 16384 --batch_size $BATCH_SIZE --seed 42 \
    --temperature 0.6 

KAT_EVAL_FOLDER=${EXPERIMENT_DIR}/kat_llm_eval

mkdir -p $KAT_EVAL_FOLDER
KAT_EVAL_FILE=${KAT_EVAL_FOLDER}/${GEN_MODEL_NAME}_kat_eval.json
## post-process KAT evaluation
python -m src.kat_post_proc \
    --extraction_folder $KAT_CHUNKS_FOLDER \
    --original_dset $KAT_DATASET_OUTPUT \
    --output_file $KAT_EVAL_FILE > $KAT_EVAL_FOLDER/log.txt

## 3 more runs to fix any mistakes by model

echo "Regenerating KAT ..."

echo "==========================" >> $KAT_EVAL_FOLDER/log.txt

python -m src.run_kat \
    --jsonl_path $KAT_DATASET_OUTPUT \
    --prompt prompts/KAT_prompt.txt \
    --output_dir $KAT_CHUNKS_FOLDER \
    --num_gpus 4 --model deepseek-r1-32b \
    --max_new_tokens 16384 --batch_size $BATCH_SIZE --seed 42 \
    --temperature 0.6 \
    --failed_file $KAT_EVAL_FILE.failed

python -m src.kat_post_proc \
    --extraction_folder $KAT_CHUNKS_FOLDER \
    --original_dset $KAT_DATASET_OUTPUT \
    --output_file $KAT_EVAL_FILE >> $KAT_EVAL_FOLDER/log.txt



echo "==========================" >> $KAT_EVAL_FOLDER/log.txt

python -m src.run_kat \
    --jsonl_path $KAT_DATASET_OUTPUT \
    --prompt prompts/KAT_prompt.txt \
    --output_dir $KAT_CHUNKS_FOLDER \
    --num_gpus 4 --model deepseek-r1-32b \
    --max_new_tokens 16384 --batch_size $BATCH_SIZE --seed 42 \
    --temperature 0.6 \
    --failed_file $KAT_EVAL_FILE.failed

python -m src.kat_post_proc \
    --extraction_folder $KAT_CHUNKS_FOLDER \
    --original_dset $KAT_DATASET_OUTPUT \
    --output_file $KAT_EVAL_FILE >> $KAT_EVAL_FOLDER/log.txt


echo "==========================" >> $KAT_EVAL_FOLDER/log.txt

python -m src.run_kat \
    --jsonl_path $KAT_DATASET_OUTPUT \
    --prompt prompts/KAT_prompt.txt \
    --output_dir $KAT_CHUNKS_FOLDER \
    --num_gpus 4 --model deepseek-r1-32b \
    --max_new_tokens 16384 --batch_size $BATCH_SIZE --seed 42 \
    --temperature 0.6 \
    --failed_file $KAT_EVAL_FILE.failed

python -m src.kat_post_proc \
    --extraction_folder $KAT_CHUNKS_FOLDER \
    --original_dset $KAT_DATASET_OUTPUT \
    --output_file $KAT_EVAL_FILE >> $KAT_EVAL_FOLDER/log.txt

## report all analysis
python -m src.evaluate.compute_all_metrics \
    $KAT_DATASET_OUTPUT \
    $KAT_EVAL_FILE \
    $DATASET_NAME \
    $EXPERIMENT_DIR/kat_eval_report.json > $EXPERIMENT_DIR/EVALUATION_REPORT.txt

CUDA_VISIBLE_DEVICES=0 python -m src.evaluate.unieval_bleu_rouge \
    --generations $GENERATION_OTUPUT \
    --dataset $DATASET_NAME/opendialkg.csv > $EXPERIMENT_DIR/unieval_results_Normal.txt & \
CUDA_VISIBLE_DEVICES=1 python -m src.evaluate.unieval_bleu_rouge \
    --generations $GENERATION_OTUPUT \
    --dataset $DATASET_NAME/opendialkg.csv --deanonymize > $EXPERIMENT_DIR/unieval_results_deanon.txt & wait

echo "Done!"

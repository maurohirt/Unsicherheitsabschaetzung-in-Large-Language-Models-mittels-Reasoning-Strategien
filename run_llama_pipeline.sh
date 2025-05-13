#This file is not used in slurm scripts its just for reference how the authors run the pipeline 

export PYTHONPATH=./
MODEL_ENGINE=$1
UQ_ENGINE=$2
DATASET=$3
OUTPUT_PATH=$4

TEMP=1.0
TRY_TIMES=20

python inference_refining.py --dataset ${DATASET} --model_engine ${MODEL_ENGINE} --model_path ${MODEL_ENGINE} --temperature ${TEMP} --output_path ${OUTPUT_PATH} --try_times ${TRY_TIMES}


python stepuq.py --dataset ${DATASET} --uq_engine ${UQ_ENGINE} --model_path ${MODEL_ENGINE} --temperature ${TEMP} --output_path ${OUTPUT_PATH} --try_times 5
#!/bin/bash 
#SBATCH -p v100 -N 1  --cpus-per-gpu=4  --gpus-per-node=1 

LAMBDA_PARAM=1
CORPUS_FILE=data/bar/md1_new.csv
S_COLUMNS=s2,s5,s7,s8
INPUT_SIZE=4

python filtered_multiples_of_frames.py $LAMBDA_PARAM 200000
# python train_lstm_kan.py --lambda_param 15 --corpusFile data/bar/md15_new.csv  --s_columns s1,s2,s5 --input_size 3
# python filtered_multiples_of_frames.py $LAMBDA_PARAM 200000
python train_lstm_kan.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python evaluate_kan.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python huitu_kan_lstm.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python convert2xvg.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE 
##--cpus-per-gpu=4  --gpus-per-node=1


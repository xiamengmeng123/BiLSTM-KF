#!/bin/bash 
#SBATCH -p v100 -N 1  --cpus-per-gpu=4  --gpus-per-node=1 

LAMBDA_PARAM=0
CORPUS_FILE=data/bar/md0_new.csv
S_COLUMNS=s2,s6,s7
INPUT_SIZE=3

python filtered_multiples_of_frames.py $LAMBDA_PARAM 200000
# python train_lstm_kan.py --lambda_param 15 --corpusFile data/bar/md15_new.csv  --s_columns s1,s2,s5 --input_size 3
# python filtered_multiples_of_frames.py $LAMBDA_PARAM 200000
python train_lstm_kan.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python evaluate_kan.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python huitu_kan_lstm.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python convert2xvg.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE 
##--cpus-per-gpu=4  --gpus-per-node=1





LAMBDA_PARAM=1
CORPUS_FILE=data/bar/md1_new.csv
S_COLUMNS=s2,s5,s7,s8
INPUT_SIZE=4

python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
# python train_lstm_kan.py --lambda_param 15 --corpusFile data/bar/md15_new.csv  --s_columns s1,s2,s5 --input_size 3
# python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
python train_lstm_kan.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python evaluate_kan.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python huitu_kan_lstm.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python convert2xvg.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE 
##--cpus-per-gpu=4  --gpus-per-node=1

 


LAMBDA_PARAM=2
CORPUS_FILE=data/bar/md2_new.csv
S_COLUMNS=s2,s5,s7,s8
INPUT_SIZE=4

python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
# python train_lstm_kan.py --lambda_param 15 --corpusFile data/bar/md15_new.csv  --s_columns s1,s2,s5 --input_size 3
# python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
python train_lstm_kan.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python evaluate_kan.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python huitu_kan_lstm.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python convert2xvg.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE 
##--cpus-per-gpu=4  --gpus-per-node=1

 


LAMBDA_PARAM=3
CORPUS_FILE=data/bar/md3_new.csv
S_COLUMNS=s2,s5,s7,s8
INPUT_SIZE=4

python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
# python train_lstm_kan.py --lambda_param 15 --corpusFile data/bar/md15_new.csv  --s_columns s1,s2,s5 --input_size 3
# python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
python train_lstm_kan.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python evaluate_kan.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python huitu_kan_lstm.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python convert2xvg.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE 
##--cpus-per-gpu=4  --gpus-per-node=1

 


LAMBDA_PARAM=4
CORPUS_FILE=data/bar/md4_new.csv
S_COLUMNS=s2,s5,s7,s8
INPUT_SIZE=4

python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
# python train_lstm_kan.py --lambda_param 15 --corpusFile data/bar/md15_new.csv  --s_columns s1,s2,s5 --input_size 3
# python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
python train_lstm_kan.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python evaluate_kan.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python huitu_kan_lstm.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python convert2xvg.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE 
##--cpus-per-gpu=4  --gpus-per-node=1

 


LAMBDA_PARAM=5
CORPUS_FILE=data/bar/md5_new.csv
S_COLUMNS=s2,s5,s7,s8
INPUT_SIZE=4

python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
# python train_lstm_kan.py --lambda_param 15 --corpusFile data/bar/md15_new.csv  --s_columns s1,s2,s5 --input_size 3
# python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
python train_lstm_kan.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python evaluate_kan.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python huitu_kan_lstm.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python convert2xvg.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE 
##--cpus-per-gpu=4  --gpus-per-node=1

 


LAMBDA_PARAM=6
CORPUS_FILE=data/bar/md6_new.csv
S_COLUMNS=s2,s5,s7,s8
INPUT_SIZE=4

python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
# python train_lstm_kan.py --lambda_param 15 --corpusFile data/bar/md15_new.csv  --s_columns s1,s2,s5 --input_size 3
# python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
python train_lstm_kan.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python evaluate_kan.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python huitu_kan_lstm.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python convert2xvg.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE 
##--cpus-per-gpu=4  --gpus-per-node=1

 


LAMBDA_PARAM=7
CORPUS_FILE=data/bar/md7_new.csv
S_COLUMNS=s2,s5,s7,s8
INPUT_SIZE=4

python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
# python train_lstm_kan.py --lambda_param 15 --corpusFile data/bar/md15_new.csv  --s_columns s1,s2,s5 --input_size 3
# python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
python train_lstm_kan.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python evaluate_kan.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python huitu_kan_lstm.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python convert2xvg.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE 
##--cpus-per-gpu=4  --gpus-per-node=1

 


LAMBDA_PARAM=8
CORPUS_FILE=data/bar/md8_new.csv
S_COLUMNS=s2,s5,s7,s8
INPUT_SIZE=4

python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
# python train_lstm_kan.py --lambda_param 15 --corpusFile data/bar/md15_new.csv  --s_columns s1,s2,s5 --input_size 3
# python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
python train_lstm_kan.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python evaluate_kan.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python huitu_kan_lstm.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python convert2xvg.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE 
##--cpus-per-gpu=4  --gpus-per-node=1

 


LAMBDA_PARAM=9
CORPUS_FILE=data/bar/md9_new.csv
S_COLUMNS=s2,s5,s7,s8
INPUT_SIZE=4

python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
# python train_lstm_kan.py --lambda_param 15 --corpusFile data/bar/md15_new.csv  --s_columns s1,s2,s5 --input_size 3
# python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
python train_lstm_kan.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python evaluate_kan.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python huitu_kan_lstm.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python convert2xvg.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE 
##--cpus-per-gpu=4  --gpus-per-node=1

 


LAMBDA_PARAM=10
CORPUS_FILE=data/bar/md10_new.csv
S_COLUMNS=s2,s5,s7,s8
INPUT_SIZE=4

python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
# python train_lstm_kan.py --lambda_param 15 --corpusFile data/bar/md15_new.csv  --s_columns s1,s2,s5 --input_size 3
# python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
python train_lstm_kan.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python evaluate_kan.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python huitu_kan_lstm.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python convert2xvg.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE 
##--cpus-per-gpu=4  --gpus-per-node=1

 


LAMBDA_PARAM=11
CORPUS_FILE=data/bar/md11_new.csv
S_COLUMNS=s2,s5,s7,s8
INPUT_SIZE=4

python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
# python train_lstm_kan.py --lambda_param 15 --corpusFile data/bar/md15_new.csv  --s_columns s1,s2,s5 --input_size 3
# python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
python train_lstm_kan.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python evaluate_kan.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python huitu_kan_lstm.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python convert2xvg.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE 
##--cpus-per-gpu=4  --gpus-per-node=1

 


LAMBDA_PARAM=12
CORPUS_FILE=data/bar/md12_new.csv
S_COLUMNS=s2,s5,s7,s8
INPUT_SIZE=4

python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
# python train_lstm_kan.py --lambda_param 15 --corpusFile data/bar/md15_new.csv  --s_columns s1,s2,s5 --input_size 3
# python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
python train_lstm_kan.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python evaluate_kan.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python huitu_kan_lstm.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python convert2xvg.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE 
##--cpus-per-gpu=4  --gpus-per-node=1

 


LAMBDA_PARAM=13
CORPUS_FILE=data/bar/md13_new.csv
S_COLUMNS=s2,s5,s7,s8
INPUT_SIZE=4

python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
# python train_lstm_kan.py --lambda_param 15 --corpusFile data/bar/md15_new.csv  --s_columns s1,s2,s5 --input_size 3
# python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
python train_lstm_kan.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python evaluate_kan.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python huitu_kan_lstm.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python convert2xvg.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE 
##--cpus-per-gpu=4  --gpus-per-node=1

 


LAMBDA_PARAM=14
CORPUS_FILE=data/bar/md14_new.csv
S_COLUMNS=s2,s5,s7,s8
INPUT_SIZE=4

python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
# python train_lstm_kan.py --lambda_param 15 --corpusFile data/bar/md15_new.csv  --s_columns s1,s2,s5 --input_size 3
# python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
python train_lstm_kan.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python evaluate_kan.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python huitu_kan_lstm.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python convert2xvg.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE 
##--cpus-per-gpu=4  --gpus-per-node=1

 


LAMBDA_PARAM=15
CORPUS_FILE=data/bar/md15_new.csv
S_COLUMNS=s2,s5,s7,s8
INPUT_SIZE=4

python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
# python train_lstm_kan.py --lambda_param 15 --corpusFile data/bar/md15_new.csv  --s_columns s1,s2,s5 --input_size 3
# python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
python train_lstm_kan.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python evaluate_kan.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python huitu_kan_lstm.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python convert2xvg.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE 
##--cpus-per-gpu=4  --gpus-per-node=1

 


LAMBDA_PARAM=16
CORPUS_FILE=data/bar/md16_new.csv
S_COLUMNS=s2,s5,s7,s8
INPUT_SIZE=4

python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
# python train_lstm_kan.py --lambda_param 15 --corpusFile data/bar/md15_new.csv  --s_columns s1,s2,s5 --input_size 3
# python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
python train_lstm_kan.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python evaluate_kan.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python huitu_kan_lstm.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python convert2xvg.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE 
##--cpus-per-gpu=4  --gpus-per-node=1

 


LAMBDA_PARAM=17
CORPUS_FILE=data/bar/md17_new.csv
S_COLUMNS=s2,s5,s7,s8
INPUT_SIZE=4

python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
# python train_lstm_kan.py --lambda_param 15 --corpusFile data/bar/md15_new.csv  --s_columns s1,s2,s5 --input_size 3
# python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
python train_lstm_kan.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python evaluate_kan.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python huitu_kan_lstm.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python convert2xvg.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE 
##--cpus-per-gpu=4  --gpus-per-node=1

 


LAMBDA_PARAM=18
CORPUS_FILE=data/bar/md18_new.csv
S_COLUMNS=s2,s5,s7,s8
INPUT_SIZE=4

python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
# python train_lstm_kan.py --lambda_param 15 --corpusFile data/bar/md15_new.csv  --s_columns s1,s2,s5 --input_size 3
# python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
python train_lstm_kan.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python evaluate_kan.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python huitu_kan_lstm.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python convert2xvg.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE 
##--cpus-per-gpu=4  --gpus-per-node=1

 


LAMBDA_PARAM=19
CORPUS_FILE=data/bar/md19_new.csv
S_COLUMNS=s2,s5,s7,s8
INPUT_SIZE=4

python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
# python train_lstm_kan.py --lambda_param 15 --corpusFile data/bar/md15_new.csv  --s_columns s1,s2,s5 --input_size 3
# python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
python train_lstm_kan.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python evaluate_kan.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python huitu_kan_lstm.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python convert2xvg.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE 
##--cpus-per-gpu=4  --gpus-per-node=1

 


LAMBDA_PARAM=20
CORPUS_FILE=data/bar/md20_new.csv
S_COLUMNS=s2,s5,s7
INPUT_SIZE=3

python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
# python train_lstm_kan.py --lambda_param 15 --corpusFile data/bar/md15_new.csv  --s_columns s1,s2,s5 --input_size 3
# python filtered_multiples_of_frames.py $LAMBDA_PARAM 50000
python train_lstm_kan.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python evaluate_kan.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python huitu_kan_lstm.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
python convert2xvg.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE 
##--cpus-per-gpu=4  --gpus-per-node=1


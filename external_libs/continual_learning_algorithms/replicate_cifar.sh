ARCH='RESNET-S'
NUM_RUNS=1
BATCH_SIZE=10
EPS_MEM_BATCH_SIZE=10

OPTIM='SGD'
lr=0.03
lam=0.0
LOG_DIR='results/cifar100'
if [ ! -d $LOG_DIR ]; then
    mkdir -pv $LOG_DIR
fi

python3 ./conv_split_cifar.py --train-single-epoch --mem-size 1 --arch $ARCH --random-seed 1234 --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --learning-rate $lr --imp-method 'A-GEM' --synap-stgth 0.0 --log-dir $LOG_DIR
python3 ./conv_split_cifar.py --train-single-epoch --mem-size 1 --arch $ARCH --random-seed 7295 --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --learning-rate $lr --imp-method 'A-GEM' --synap-stgth 0.0 --log-dir $LOG_DIR
python3 ./conv_split_cifar.py --train-single-epoch --mem-size 1 --arch $ARCH --random-seed 5234 --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --learning-rate $lr --imp-method 'A-GEM' --synap-stgth 0.0 --log-dir $LOG_DIR

python3 ./conv_split_cifar.py --train-single-epoch --mem-size 1 --arch $ARCH --random-seed 1234 --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --learning-rate $lr --imp-method 'EWC' --synap-stgth 10.0 --log-dir $LOG_DIR
python3 ./conv_split_cifar.py --train-single-epoch --mem-size 1 --arch $ARCH --random-seed 7295 --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --learning-rate $lr --imp-method 'EWC' --synap-stgth 10.0 --log-dir $LOG_DIR
python3 ./conv_split_cifar.py --train-single-epoch --mem-size 1 --arch $ARCH --random-seed 5234 --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --learning-rate $lr --imp-method 'EWC' --synap-stgth 10.0 --log-dir $LOG_DIR


python3 ./conv_split_cifar.py --train-single-epoch --mem-size 1 --arch $ARCH --random-seed 1234 --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --learning-rate $lr --imp-method 'ER' --synap-stgth 0.0 --log-dir $LOG_DIR
python3 ./conv_split_cifar.py --train-single-epoch --mem-size 1 --arch $ARCH --random-seed 7295 --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --learning-rate $lr --imp-method 'ER' --synap-stgth 0.0 --log-dir $LOG_DIR
python3 ./conv_split_cifar.py --train-single-epoch --mem-size 1 --arch $ARCH --random-seed 5234 --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --learning-rate $lr --imp-method 'ER' --synap-stgth 0.0 --log-dir $LOG_DIR


echo "************************ replicating experiment 2 (rotated MNIST) ***********************"
echo " >>>>>>>> Plastic (Naive) SGD "
python -m stable_sgd.main --dataset rot-mnist --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --hiddens 256 --batch-size 10 --dropout 0.0 --seed 1234
python -m stable_sgd.main --dataset rot-mnist --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --hiddens 256 --batch-size 10 --dropout 0.0 --sedd 4567
python -m stable_sgd.main --dataset rot-mnist --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --hiddens 256 --batch-size 10 --dropout 0.0 --seed 7891

echo ""
echo " >>>>>>>> Stable SGD "
python -m stable_sgd.main --dataset rot-mnist --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 0.5 --hiddens 256 --batch-size 10 --dropout 0.5 --seed 1234
python -m stable_sgd.main --dataset rot-mnist --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 0.5 --hiddens 256 --batch-size 10 --dropout 0.5 --sedd 4567
python -m stable_sgd.main --dataset rot-mnist --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 0.5 --hiddens 256 --batch-size 10 --dropout 0.5 --seed 7891

echo ""
echo ">>>>>>>>> Other Methods (ER, A-GEM, EWC)"
echo "Make sure you have tensorflow==1.12 installed. (see the readme doc)"
cd ./external_libs/continual_learning_algorithms
bash replicate_mnist.sh rot-mnist
cd ../..

echo "************************ replicating experiment 2 (permuted MNIST) ***********************"
echo " >>>>>>>> Plastic (Naive) SGD "
python -m stable_sgd.main --dataset perm-mnist --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --hiddens 256 --batch-size 10 --dropout 0.0 --seed 1234
python -m stable_sgd.main --dataset perm-mnist --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --hiddens 256 --batch-size 10 --dropout 0.0 --seed 4567
python -m stable_sgd.main --dataset perm-mnist --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --hiddens 256 --batch-size 10 --dropout 0.0 --seed 7891

echo ""
echo " >>>>>>>> Stable SGD "
python -m stable_sgd.main --dataset perm-mnist --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 0.8 --hiddens 256 --batch-size 10 --dropout 0.5 --seed 1234
python -m stable_sgd.main --dataset perm-mnist --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 0.8 --hiddens 256 --batch-size 10 --dropout 0.5 --seed 4567
python -m stable_sgd.main --dataset perm-mnist --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 0.8 --hiddens 256 --batch-size 10 --dropout 0.5 --seed 7891

echo ""
echo ">>>>>>>>> Other Methods (ER, A-GEM, EWC)"
echo "Make sure you have tensorflow==1.12 installed. (see the readme doc)"
cd ./external_libs/continual_learning_algorithms
bash replicate_mnist.sh perm-mnist
cd ../..



echo "************************ replicating experiment 2 (Split CIFAR-100) ***********************"
echo " >>>>>>>> Plastic (Naive) SGD "
python -m stable_sgd.main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --hiddens 256 --batch-size 10 --dropout 0.0 --seed 1234
python -m stable_sgd.main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --hiddens 256 --batch-size 10 --dropout 0.0 --seed 4567
python -m stable_sgd.main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --hiddens 256 --batch-size 10 --dropout 0.0 --seed 7891

echo ""
echo " >>>>>>>> Stable SGD "
python -m stable_sgd.main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 0.8 --hiddens 256 --batch-size 10 --dropout 0.5 --seed 1234
python -m stable_sgd.main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 0.8 --hiddens 256 --batch-size 10 --dropout 0.5 --seed 4567
python -m stable_sgd.main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 0.8 --hiddens 256 --batch-size 10 --dropout 0.5 --seed 7891

echo ""
echo ">>>>>>>>> Other Methods (ER, A-GEM, EWC)"
echo "Make sure you have tensorflow==1.12 installed. (see the readme doc)"
cd ./external_libs/continual_learning_algorithms
bash replicate_cifar.sh

echo "************************ replicating experiment 1 (rotated MNIST) ***********************"
echo " >>>>>>>> Plastic (Naive) SGD "
python -m stable_sgd.main --dataset rot-mnist --tasks 5 --epochs-per-task 5 --lr 0.01 --gamma 1.0 --hiddens 100 --batch-size 64 --dropout 0.0 --seed 1234
python -m stable_sgd.main --dataset rot-mnist --tasks 5 --epochs-per-task 5 --lr 0.01 --gamma 1.0 --hiddens 100 --batch-size 64 --dropout 0.0 --seed 4567
python -m stable_sgd.main --dataset rot-mnist --tasks 5 --epochs-per-task 5 --lr 0.01 --gamma 1.0 --hiddens 100 --batch-size 64 --dropout 0.0 --seed 7891

echo " >>>>>>>> Stable SGD "
python -m stable_sgd.main --dataset rot-mnist --tasks 5 --epochs-per-task 5 --lr 0.1 --gamma 0.4 --hiddens 100 --batch-size 16 --dropout 0.5 --seed 1234
python -m stable_sgd.main --dataset rot-mnist --tasks 5 --epochs-per-task 5 --lr 0.1 --gamma 0.4 --hiddens 100 --batch-size 16 --dropout 0.5 --seed 4567
python -m stable_sgd.main --dataset rot-mnist --tasks 5 --epochs-per-task 5 --lr 0.1 --gamma 0.4 --hiddens 100 --batch-size 16 --dropout 0.5 --seed 7891



echo "************************ replicating experiment 1 (permuted MNIST) ***********************"
echo " >>>>>>>> Plastic (Naive) SGD "
python -m stable_sgd.main --dataset perm-mnist --tasks 5 --epochs-per-task 5 --lr 0.01 --gamma 1.0 --hiddens 100 --batch-size 64 --dropout 0.0 --seed 1234
python -m stable_sgd.main --dataset perm-mnist --tasks 5 --epochs-per-task 5 --lr 0.01 --gamma 1.0 --hiddens 100 --batch-size 64 --dropout 0.0 --seed 4567
python -m stable_sgd.main --dataset perm-mnist --tasks 5 --epochs-per-task 5 --lr 0.01 --gamma 1.0 --hiddens 100 --batch-size 64 --dropout 0.0 --seed 7891

echo " >>>>>>>> Stable SGD "
python -m stable_sgd.main --dataset perm-mnist --tasks 5 --epochs-per-task 5 --lr 0.1 --gamma 0.4 --hiddens 100 --batch-size 16 --dropout 0.5 --seed 1234
python -m stable_sgd.main --dataset perm-mnist --tasks 5 --epochs-per-task 5 --lr 0.1 --gamma 0.4 --hiddens 100 --batch-size 16 --dropout 0.5 --seed 4567
python -m stable_sgd.main --dataset perm-mnist --tasks 5 --epochs-per-task 5 --lr 0.1 --gamma 0.4 --hiddens 100 --batch-size 16 --dropout 0.5 --seed 7891

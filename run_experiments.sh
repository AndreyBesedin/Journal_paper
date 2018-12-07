#ROOT=/home/besedin/workspace/Projects/Journal_paper/
ROOT=/data/DEEPLEARNING/besed_an/workspace/Projects/Journal_paper/
#ROOT=/home/abesedin/workspace/Projects/Journal_paper/
#batch classification

#python batch_classification_training.py --dataset MNIST --root $ROOT --cuda --batch_size 1000 --lr 0.001 --optimizer Adam --cuda_device 0 --niter 25
python batch_classification_training.py --dataset LSUN --root $ROOT --cuda --batch_size 1000 --lr 0.001 --optimizer SGD --cuda_device 0 --niter 25
#python batch_classification_training.py --dataset Synthetic --root $ROOT --cuda --batch_size 1000 --lr 0.001 --optimizer SGD --cuda_device 0 --niter 25

#python representativity_test.py --dataset MNIST --root /home/besedin/workspace/Projects/Journal_paper/ --betta1 0.02 --cuda --batch_size 100 --lr 0.001
#python representativity_test.py --dataset LSUN --root $ROOT --betta1 0 --cuda --cuda_device 0 --batch_size 100 --lr 0.001 --optimizer SGD --code_size 32


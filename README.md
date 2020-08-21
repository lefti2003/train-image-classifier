# train-image-classifier
program to train an image classifier
This is a python script that will train a new network on a dataset and save the model as a checkpoint.
# Usage
1. (Basic):python train.py data_directory
2. (Save checkpoints) :python train.py data_directory --save_dir save_directory
3. (Choose architecture):python train.py data_dir --arch "vgg13"
4. (Set hyperparameters):python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
5. (Use GPU for training):python train.py data_dir --gpu

# KDD2021

This is the source code of ATM-GCN and the recommendation GCN model. 

Requirements:  
Pytorch 10.1  
python 3.7  
CUDA 10.1  

Data download address:  
https://grouplens.org/datasets/movielens/  
https://www.kaggle.com/c/kkbox-music-recommendation-challenge/data  

Pre-training data format: (no space around tab)   
sub \t rel type \t obj \t rel value  
Recommendation data format: (no space around tab)  
user_id \t item_id  

In ATM-GCN folder, run the command for pre-training  
python run.py -data movielens -score_func transe -opn sub -gamma 9 -hid_drop 0.1 -init_dim 60  
The learned node embeddings will be in the 'checkpoints' folder.

Then in CF folder, run the following code for recommendation  
python main.py --dataset movielens --embed_size 60 --layer_size [60,60,60] --lr 0.0001 --save_flag 1 --batch_size 1000 --epoch 400 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1]  --load_path [checkpoint file name]



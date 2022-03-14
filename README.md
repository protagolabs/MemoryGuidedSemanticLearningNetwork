# Data Preparation

Word embedding: http://nlp.stanford.edu/data/glove.840B.300d.zip

ActivityNet: http://activity-net.org/challenges/2016/download.html

TACoS: https://drive.google.com/file/d/1kK_FTo6USmPhO1vam3uvBMtJ3QChUblm/view

Charades: https://app.box.com/s/h0sxa5klco6qve5ahnz50ly2nksmuedw


# Training

ActivityNet: python main.py --word2vec-path /yourpath/glove_model.bin --dataset ActivityNet --feature-path /yourpath/ActivityCaptions/ActivityC3D --train-data data/activity/train_data_gcn.json --val-data data/activity/val_data_gcn.json --test-data data/activity/test_data_gcn.json --max-num-epochs 20 --dropout 0.2 --warmup-updates 300 --warmup-init-lr 1e-06 --lr 8e-4 --num-heads 4 --num-gcn-layers 2 --num-attn-layers 2 --weight-decay 1e-7 --train --model-saved-path models_activity

TACoS: python main.py --word2vec-path /yourpath/glove_model.bin --dataset TACOS --feature-path /yourpath/TACOS/TACOS --train-data data/tacos/TACOS_train_gcn.json --val-data data/tacos/TACOS_val_gcn.json --test-data data/tacos/TACOS_test_gcn.json --max-num-epochs 60 --dropout 0.2 --warmup-updates 300 --warmup-init-lr 1e-07 --lr 4e-4 --num-heads 4 --num-gcn-layers 2 --num-attn-layers 2 --weight-decay 1e-8 --train --model-saved-path models_tacos --batch-size 64

Charades: python main.py --word2vec-path /yourpath/glove_model.bin --dataset Charades --feature-path /yourpath/Charades --train-data data/charades/train.json --val-data data/charades/test.json --test-data data/charades/test.json --max-num-epochs 80 --dropout 0.2 --warmup-updates 300 --warmup-init-lr 1e-07 --lr 4e-4 --num-heads 4 --num-gcn-layers 2 --num-attn-layers 2 --weight-decay 1e-8 --train --model-saved-path models_charades --batch-size 64 --max-num-frames 64


# Testing

ActivityNet: python main.py --word2vec-path /yourpath/glove_model.bin --dataset ActivityNet --feature-path /yourpath/ActivityCaptions/ActivityC3D --train-data data/activity/train_data_gcn.json --val-data data/activity/val_data_gcn.json --test-data data/activity/test_data_gcn.json --max-num-epochs 20 --dropout 0.2 --warmup-updates 300 --warmup-init-lr 1e-06 --lr 8e-4 --num-heads 4 --num-gcn-layers 2 --num-attn-layers 2 --weight-decay 1e-7 --evaluate --model-load-path /your/model/path

TACoS: python main.py --word2vec-path /yourpath/glove_model.bin --dataset TACOS --feature-path /yourpath/TACOS/TACOS --train-data data/tacos/TACOS_train_gcn.json --val-data data/tacos/TACOS_val_gcn.json --test-data data/tacos/TACOS_test_gcn.json --max-num-epochs 40 --dropout 0.2 --warmup-updates 300 --warmup-init-lr 1e-07 --lr 4e-4 --num-heads 4 --num-gcn-layers 2 --num-attn-layers 2 --weight-decay 1e-8 --evaluate --batch-size 64 --model-load-path /your/model/path

Charades: python main.py --word2vec-path /yourpath/glove_model.bin --dataset Charades --feature-path /yourpath/Charades --train-data data/charades/train.json --val-data data/charades/test.json --test-data data/charades/test.json --max-num-epochs 40 --dropout 0.2 --warmup-updates 300 --warmup-init-lr 1e-07 --lr 4e-4 --num-heads 4 --num-gcn-layers 2 --num-attn-layers 2 --weight-decay 1e-8 --evaluate --batch-size 64 --max-num-frames 64 --model-load-path /your/model/path

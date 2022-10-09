# Dacon Super-Resolution

## Dataset
1.Clone this repository
```
git clone https://github.com/onebottlekick/dacon_super_resolution.git
cd dacon_sr
```
2.Download dataset
```
download and place dataset in "dataset" dir.
```
3.extract dataset
```
cd dataset
python extract.py
```
dataset dir should be like this
```
dataset
|   data_augmentation.py
|   dataloader.py
│   dataset.py
|   extract.py
|   open.zip
|   preprocess.py  
|   ...  
│
└───train
|   |
│   └───hr
|   |   |   0001.png
|   |   |   0002.png
|   |   |   ...
│   │
│   └───lr
│       │   0000.png
│       │   0001.png
│       │   ...
│   
└───test
    |
    └───lr
        |   20000.png
        |   20001.png
        |   ...
```
4.make patches
```
python preprocess.py
```

## Train
```
python main.py --save-dir ./train/Model \
               --log-file-name train.log \
               --reset \
               --gpu \
               --hr-dataset-dir ./dataset/hr_patches \
               --lr-dataset-dir ./dataset/lr_patches \
               --train-val-split 0.8 \
               --num-blocks 3 \
               --num-local-blocks 3 \
               --num-features 64 \
               --reconstruction-loss-type mse \
               --batch-size 16 \
               --num-epochs 1000 \
               --print-every 300 \
               --save-every 1 \
               --val-every 20
```

## Evaluate
```
python main.py --save-dir ./eval/Model \
               --log-file-name eval.log \
               --reset \
               --gpu \
               --eval \
               --eval-save-results \
               --train-val-split 0.99 \
               --num-blocks 3 \
               --num-local-blocks 3 \
               --num-features 64 \
               --checkpoint-path $(pretrained_model_path)
```

## Test
```
python main.py --save-dir ./test/Model \
               --log-file-name test.log \
               --reset \
               --gpu \
               --test \
               --test-img-path ./dataset/test/lr \
               --num-blocks 3 \
               --num-local-blocks 3 \
               --num-features 64 \
               --checkpoint-path $(pretrained_model_path)
```
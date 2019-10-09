# python3 train.py --img-size 1024 --transfer --batch-size 4
# python train.py --img-size 416 --transfer --batch-size 4 --freeze --save_per_epoch 10 --multi-scale
python train.py --img-size 416 --transfer --batch-size 4 --freeze --save_per_epoch 10 --multi-scale --cfg cfg/soccer.cfg  --data-cfg data/soccer.data
# python3 train.py --img-size 1024 --transfer --batch-size 4 --save_per_epoch 10 --data-cfg data/coco2.data

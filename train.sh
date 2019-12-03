# python3 train.py --img-size 1024 --transfer --batch-size 4
# python train.py --img-size 416 --transfer --batch-size 4 --freeze --save_per_epoch 10 --multi-scale --cfg cfg/soccer.cfg  --data-cfg data/soccer.data
# python train.py --img-size 416 --transfer --batch-size 4 --freeze --save_per_epoch 10 --multi-scale
# python3 train.py --img-size 1024 --transfer --batch-size 4 --save_per_epoch 10 --data-cfg data/coco2.data
python train.py --img-size 416 --transfer --batch-size 4 \
       --save_per_epoch 1 --multi-scale --cfg cfg/soccer2.cfg  \
       --data-cfg data/soccer.data

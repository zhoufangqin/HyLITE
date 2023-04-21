# LAHIT

## example of training
$python main.py --dataset='Indian' --epoches=300 --patches=7 --band_patches=1 --mode='CAF' --weight_decay=5e-3 --flag='train' --output_dir='./logs/' --batch_size=32 --align='align' --spatial_attn

## example of visualizing the results
python visualization.py

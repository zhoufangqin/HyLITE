# Locality-Aware Hyperspectral Classification
The code implements the ["Locality-Aware Hyperspectral Classification (BMVC2023)"](https://arxiv.org/pdf/2309.01561.pdf)
![image](https://github.com/zhoufangqin/HyLITE/blob/main/Architecture_v4.png)

## Example of training
$python main.py --dataset='Indian' --epoches=300 --patches=7 --band_patches=1 --mode='CAF' --weight_decay=5e-3 --flag='train' --output_dir='./logs/' --batch_size=32 --align='align' --spatial_attn

## Example of visualizing the results
$python visualization.py

## Experimental setup
For a detailed experimental setup and dataset information, please refer to our [supplementary materials](https://bmvc2022.mpi-inf.mpg.de/BMVC2023/0022_supp.pdf).

## Citations:
The code is built upon [SpectralFormer](https://github.com/danfenghong/IEEE_TGRS_SpectralFormer#spectralformer-rethinking-hyperspectral-image-classification-with-transformers)
 and [MAEST](https://github.com/ibanezfd/MAEST), thanks to their great work! If you find it is useful for your research, please kindly cite the following papers:

- [Zhou et al. (2023)](https://arxiv.org/pdf/2309.01561.pdf) - Locality-Aware Hyperspectral Classification
- [Hong et al. (2021)](https://ieeexplore.ieee.org/document/9627165). - SpectralFormer: Rethinking Hyperspectral Image Classification With Transformers
- [Damian et al. (2022)](https://ieeexplore.ieee.org/document/9931741) - Masked Auto-Encoding Spectral–Spatial Transformer for Hyperspectral Image Classification


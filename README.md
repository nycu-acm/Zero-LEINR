# Zero-LEINR: Zero-Reference Low-light Image Enhancement with Intrinsic Noise Reduction

**Wing Ho Tang**, **Hsuan Yuan**, **Tzu-Hao Chiang**, **HChing-Chun Huang**

**Official Pytorch implementation for the paper accepted by [ISCAS 2023](https://ieeexplore.ieee.org/document/10181743).**

## Python Requirements

This code was tested on:

- Python 3.8
- Pytorch 1.13

## Training

To train a network, run:

```bash
python train.py 
--train_dirs=/path/to/train/dirs 
--val_input_dirs=/path/to/val/input/dirs 
--val_gt_dirs=/path/to/val/gt/dirs 
--test_dirs=/path/to/test/dirs 
```

## Citations

```
@INPROCEEDINGS{10181743,
    author={Tang, Wing Ho and Yuan, Hsuan and Chiang, Tzu-Hao and Huang, Ching-Chun},
    booktitle={2023 IEEE International Symposium on Circuits and Systems (ISCAS)}, 
    title={Zero-LEINR: Zero-Reference Low-light Image Enhancement with Intrinsic Noise Reduction}, 
    year={2023},
    volume={},
    number={},
    pages={1-5},
    doi={10.1109/ISCAS46773.2023.10181743}}

```
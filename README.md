# Zero-LEINR: Zero-Reference Low-light Image Enhancement with Intrinsic Noise Reduction

**Wing Ho Tang**, **Hsuan Yuan**, **Tzu-Hao Chiang**, **HChing-Chun Huang**

**Official Pytorch implementation for the paper accepted by [ISCAS 2023](https://ieeexplore.ieee.org/document/10181743).**

# Abstract 

Zero-reference deep learning-based methods for low-light image enhancement sufficiently mitigate the difficulty of paired data collection while keeping the great generalization on various lighting conditions, but color bias and unintended intrinsic noise amplification are still issues that remain unsolved. In this paper, we propose a zero-reference end-to-end two-stage network (Zero-LEINR) for low-light image enhancement with intrinsic noise reduction. In the first stage, we introduce a Color Preservation and Light EnhancementBlock (CPLEB) that consists of a dual branch structure with different constraints to correct the brightness and preserve the correct color tone. In the second stage, Enhanced-NoiseReduction Block (ENRB) is applied to remove the intrinsic noises being enhanced during the first stage. Due to the zero-reference two-stage structure, our method can enhance the low-light image with the correct color tone on unseen datasets and reduce the intrinsic noise at the same time.

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

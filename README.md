# Variational Collaborative Generative Adversarial Network (VCGAN)
- pytorch implementation

Please kindly cite our article if you use this repository

## Requirement
1. python >= 3.6
1. pytorch >= 1.2.0
1. numpy >= 1.14.3
1. pandas >= 0.24.2

## Running
check specifications by 
```python
python vcgan.py -h
```

### Sample run (running on GPU)
pre-train with side information
```python
python vcgan.py --dir *path -a 1 -b 0.75 -e 200 --pretrain --save --gpu
```
If you want to test on CPU, simply remove --gpu.

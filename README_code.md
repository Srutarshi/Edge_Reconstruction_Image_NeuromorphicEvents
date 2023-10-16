# Event Edge Reconstruction
A PyTorch implementation of edge reconstruction of quadtree binned data using event frames at full resolution.

## Requirements
- [NumPy](https://www.numpy.org/)
- [PyTorch](https://pytorch.org/)

## Datasets
The dataset used is from the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2015 Video dataset. The distorted data was generated using private code for video compression with the goal of optimizing object tracking performance on the degraded video sequences.

## Usage

Please run the following to see the flags for the most help
```
python edge_reconstruct -h
```

### Training
Run the following basic command to train the model:
```
python edge_reconstruct --mode train
```

### Testing
Run the following basic command to test the model:
```
python edge_reconstruct --mode test
```

### Training
Run the following basic command to pass data without labels through the model:
```
python edge_reconstruct --mode run
```
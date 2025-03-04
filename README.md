# Settings: 
This code calculates influence scores for CIFAR-10 using ResNet-9.

# Running the Code: 
To prepare the initialized model: run python train_cifar.py
To calculate TRAK scores: run python main_trak.py

# Inputs:
Model parameters (see line 26 of main_trak.py)
Training samples (see line 12 of main_trak.py)
Target samples (see line 40 of main_trak.py)

# Output: 
The code computes the influence score of each training sample on each target sample. Upon execution, new results will be stored in ./trak_results/scores/, where an .npy file will contain a 50,000 × 10,000 array, corresponding to 50K training samples and 10K target samples. 

# Prerequisites:
Python==3.8.0
traker==0.1.3 
torch==1.13.1 (This version is compatible with TRAK and our server's CUDA setup. If using a different server, feel free to update to a compatible PyTorch version.)
torchvision==0.14.1

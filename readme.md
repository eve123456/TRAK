## Settings: 
This code calculates influence scores for CIFAR-10, a more complex image classification dataset than SVHN, using ResNet-9, a standard model for image classification tasks.

## Running the Code: 
To test the implementation, run the following command: python main_trak.py

## Inputs:
Model parameters (see line 26 of main_trak.py)
Training samples (see line 12 of main_trak.py)
Target samples (see line 40 of main_trak.py)

## Output: 
The code computes the influence score of each training sample on each target sample. You can find an example result here:
./trak_results/scores/scores_0129.npy -- This file contains a 50,000 × 10,000 array, corresponding to 50K training samples and 10K target samples. Upon execution, new results will be stored in ./trak_results/scores/.

## Prerequisites:
  ### Library Requirements:
    Python==3.8.0
    traker==0.1.3 (Note: The library name is Traker, not "trak," even though it is imported in the code as from trak import ...)
    torch==1.13.1 (This version is compatible with TRAK and our server's CUDA setup. If using a different server, feel free to update to a compatible PyTorch version.)
    torchvision==0.14.1
   ### Model Initialization: 
    Run the following command to prepare the initialized model: python train_cifar.py
# Contextual Invariants for Vehicle Re-identification

This repo contains the official code for 'Contextual Invariants for Vehicle Re-Identification'. Our approach is able to significantly improve mAP compared to comtemporary works without relying on complex architectures. We believe using our approach as a baseline will help ind eveloping more accurate vehicle re-identification models.

# Requirements

You will need the following:
    - PyTorch 1.2.0
    - NVIDIA Apex 
    - Torchvision 0.4.0
    - kaptan 0.5.12

Multi-GPU training is currently not supported. You will need to ensure torch recognizes only one GPU, otherwise several functions will throw NotImplementedError(). 

# Execution

1. Clone this repo.
2. To train a model, run 

    $ python main.py path\to\config.yml train

3. To test a model, run

    $ python main.py path\to\config.yml test \path\to\weights.pth

We have provided several configuration files, as well as details about configuration options in `config.md`.

# Additional details


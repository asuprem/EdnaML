# Key updates


- May 6 2023
  - Generated this file
  - **Repository updates**
    - This repository contains EdnaML, as well as various EdnaML profiles and experiments.
    - EdnaML is a framework for reproducible ML experimentation, as well as a pipeline manager for ML experiment in image, video, and text
    - EdnaML profiles rul EdnaML experiments for various paper publications
  - **EdnaML Profiles**
    - All profiles are in the `\profiles` directory, grouped into top-level sections. Following is a description of the profiles. Each entry in figure will consist of a changelog describing any changes made to the profiles.
    - **CarZam**
      - The CarZam profile directory contains the base code to implement CarZam single-branch, multi-label, and multi-branch experiments from [Colabel](https://arxiv.org/pdf/2205.10011.pdf). 
      - The Jupyter notebook for CarZam is also included in the profile directory. It contains additional instructions to properly execute the CarZam pipeline.
    - **FNC**
      - The FNC profile directory contains the base code to implement labeling and classification for COVID-19 misinformation detection
      - These tasks have additional details inside the jupyter notebook inside the directory
      - The labeling task implements KALM, as well as UFIT and ODIN. Specifically, given a dataset of tweets (linked inside the notebook), we download the dataset from Azure, identify keywords, filter out potentially irrelevant posts, cluster irrelevant posts to find overlap with relevant posts, then propagate labels from several existing classifiers into the unlabeled dataset using label integration (linked inside notebooks).
      - The classification task, given a set of classifiers, evaluates accuracy.
    - **Fakeddit**
      - This profile contains base code to generate a text classifier for the Fakeddit dataset.
      - The notebook, labeled Semantic Masking, contains code the run the EdnaML workflow to generate the text classifier. 
      - The first section of the notebook contains several variable knobs to adjust which model gets generated.
    - **MiDAS**
      - This profile contains base code to generate text classifiers for the EFND dataset from the [Generalizability paper](https://arxiv.org/pdf/2205.07154.pdf)\
      - It uses the same Semantic Masking notebook from `Fakeddit`, which is not placed here
    - **NELA**
      - This profile contains base code to generate a text classifier for the NELA misinformation dataset.
      - It uses the same Semantic Masking notebook from `Fakeddit`, which is not placed here
    - **PolitiFact**
      - This profile contains base code to generate a text classifier for the PolitiFact misinformation dataset, as well as the GossipCop misinformation dataset (since both were released together by the same paper).
      - It uses the same Semantic Masking notebook from `Fakeddit`, which is not placed here
  - **Branch Updates**
    - As the master branch, this is slowest to update (other than this file specifically)
    - Branch updates will be provided in this section, if there are any. Below is a brief list of current branches that are being worked on.
    - **devel**
      - This is the primary development branch. Changes are tested in this branch before being merged into master
      - Currently, devel is implementing Storage classes to better interface with Azure and Google Drive backends, and a database plugin to save experiment results in MongoDB
      - Updated prefixes and naming conventions for storage-specific configuration variables, to differentiate from other sections
      - Updated the class in LocalStorage to be LocalStorage, instead of AzureStorage from an earlier iteration
      - Added storage-management APIs directly to the EdnaML API, so they can be used declaratively as well as imperatively (specifically, addStorageClass and addStorageInstance)
    - **suprem-devel-mongo-metrics**
      - This branch is developing a Metrics API to track several high- and low-level metrics for experiments. 
      - Low-level metrics incldue performance benchmarks, runtimes, GPU/CPU usage.
      - High-level metrics are pipeline-specific, such as classification accuracy, f1-score, matrix norm, l2-norm of classifier weights, etc
      - High-level metrics are implemented using Torchmetrics where possible
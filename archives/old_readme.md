## Directories and Files

[`config.md`](config.md) provides overall configuration options for EdnaML

[`config-reference`](config-reference) provides detailed configuration options

[`sample-configs`](sample-configs) provides sample configuration files

[`config-full.yml`](config-full.yml) provides a sample full configuration file

[`docs`](docs) contains documentation for the codebase

# How it works

EdnaML takes in a configuration file. The configuration file sections are described in [`config.md`](config.md).

Execution requires an experiment file, and a configuration file.

**FUTURE WIP**
a single experiment module/file that takes in only a config file, a weights path, and a train/test mode to perform the experiments...


# Lore Overview

EdnaML allows for experimentaion on a  variety of ML problems. It's primary job is to simplify experiment tracking, and to provide a common framework for several types of experiments across different modalities of text, image, and video. Usually, we conduct ML experiments rather haphazardly, without quite keeping track of all our parameters and changes. Soon, we end up with a soup of configurations, codes, results, and hyperparameters strewn across notebooks, githubs, md files, and powerpoints, all of which are hard to track together.

I had the same problem when I started working on EdnaML. At the time, I had several ML experiments running concurrently, and I quickly ran out of ways to track them. I also investigated existing ML tracking approaches; Many of them were useful to track ML experiment parameters, but none tracked the overall pipeline, nor did they have flexibility in adapting multiple pipelines into a single canonical format.

So, I decided to create my own framework. The idea was to distill any ML experiment to the fundamental steps, then have a configuration file decide what parameters and classes to use at each fundamental step. This is at a higher level than just using parameters, because with EdnaML, we are also keeeping track of the specific functions used at each fundamental step. 

## Fundamental Steps

The fundamental steps are relatively self explanatory: they are simply the stages in an ML pipeline:

1. Load data
2. Preprocess data
3. Set up model
4. Load weghts into model
5. Set up loss functions
6. Set up optimizer
7. Set up schedulers
8. Train Model
    8a. Generate Batches from data
    8b. Get model predictions
    8c. Perform backward pass
    8d. Update model parameters
    8e. evaluate
9. Save model
10. Save results

Most existing frameworks only consider the model training part, and even there, they only focus on hyperparameters. As we can expect, though, all these steps are important in getting results: the specific augmentations we choose in `Step 1,2`, which pretrained weights we use in `Step 4` what parameters we use for `Steps 5,6,7` etc. In addition, the existing frameworks were useful for relatively simple problems, but not for more complex ML architecture designs. 

To that end, I designed EdnaML to take the entire pipeline into consideration.

## PyTorch Lightning

In some ways EdnaML is similar to PyTorch Lightning. So why keep working on this? Sunk cost, I suppose? Also, I am very used to it by now, and it is a very useful teaching tool. Enough said :)
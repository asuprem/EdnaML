# `TRANSFORMATION`

The following is a list of transformation parameters. These are tied to whicvever Data Generator one will be using. The next section lists supported transformations for each impemented Generator.

## Common

- `SHAPE`: The image shape to resize, if working with image or video data. **Array of ints**, e.g. `[200,200]`. 
- `NORMALIZATION_MEAN`: The image normalization mean parameter to use. **Float**, usually `0.5`
- `NORMALIZATION_STD`: The image normalization standard deviation parameter to use. **Float**, usually `0.5` 
- `NORMALIZATION_SCALE`: The image normalization scale parameter to use. Unused in code. **Float**, usually `255`
- `CHANNELS`: Number of channels in image. **Integer**, usually `3`
- `BATCH_SIZE`: The size of batches during traning. **Integer** power of 2
- `WORKERS`: The number of parallel workers for data loading. **Integers**. Usually `2`.


## Additional

- `H_FLIP`: Horizontal flipping parameter. **Probability between 0 and 1**
- `T_CROP`: Random cropping parameters. **Boolean**
- `RANDOM_ERASE`: Random erasing augmentation. **Boolean**
- `RANDOM_ERASE_VALUE`: Maximum fraction of image to random erase. **Value in [0,1]**

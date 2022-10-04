# DATAREADERS


DataReaders perform 3 tasks: 

1. They crawl a directory to collect paths to all files as well as associatated annotations
2. They build a TorchDataset for this data
3. They build a generator using TorchDataloader to yield batches

`DATAREADERS` is a dictionary. It looks like:


```
EXECUTION:
    .
    .
    DATAREADER: 
        DATAREADER: CompCars
        CRAWLER_ARGS:
            data_folder: "Data/CompCars"
            train_folder: "image"
        DATASET_ARGS: 
            pathidx: 4
            annotationidx: 3
            classificationclass: type
        GENERATOR_ARGS: {}      
        DATALOADER_ARGS: {}
```

`DATAREADER.DATAREADER` determines which datareader we will use. In turn, this determines what arguments exist for `CRAWLER_ARGS`, `DATASET_ARGS`, `GENERATOR_ARGS`, and `DATALOADER_ARGS`.

The following values for `DATAREADER.DATAREADER` are available:

- `CompCars`
- `VehicleColor`
- `VehicleID`
- `VeRi`

Subsequent parameters for each `DATAREADER.DATAREADER` is provided below.


## `CompCars`

The datareader for the CompCars dataset.

### `CRAWLER_ARGS`

The crawler extracts all paths to tuples. 2 arguments:

- `data_folder`: path to the root CompCars directory that contains the the `image` directory, e.g. `Data/CompCars`
- `train_folder`: name of the `image` directory, e.g. `image`
- `trainfile`: file that contains training images. Choose from `train.txt`, `trainlarge.txt`, `train7030.txt`
- `testfile`: file that contains testing  images. Choose from `test.txt`, `testlarge.txt`, `test7030.txt`

### DATASET_ARGS

TorchDataset stes up functions to load images from path, and well as yield a single annotation/label. We need to provide the index of the annotation, since CompCars has multiple annotations.
Each tuple of a sample is of the form: `(make, model, releasedyear, type, path/to/image)`
3 arguments:

- `pathidx`: The index where the path exists. This is `4` for CompCars
- `annotationidx`: the index where the annotation exists. If you want to use the `make` annotation, use `0`. 
- `classificationclass`: The name of the annotation used. See tuple above. If you want to use the make annotation, use `make`

### GENERATOR_ARGS

Unused

### DATALOADER_ARGS

Unused

## `VehicleColor`

The datareader for the VehicleColor dataset.

### `CRAWLER_ARGS`

The crawler extracts all paths to tuples. 2 arguments:

- `data_folder`: path to the root VehicleColor directory that contains the the `image` directory, e.g. `Data/VehicleColor`
- `train_folder`: name of the `train` directory, e.g. `train`
- `validation_folder`: name of the `val` directory, e.g. `val`
- `test_folder`: name of the `test` directory, e.g. `test`

### DATASET_ARGS

TorchDataset stes up functions to load images from path, and well as yield a single annotation/label. We need to provide the index of the annotation, since VehicleColor has a single annotations.
Each tuple of a sample is of the form: `(path/to/image, color)`
3 arguments:

- `pathidx`: The index where the path exists. This is `0` for VehicleColor
- `annotationidx`: the index where the annotation exists. `1` is the only option. 
- `classificationclass`: The name of the annotation used. See tuple above. `color` is the only option

### GENERATOR_ARGS

Unused

### DATALOADER_ARGS

Unused


## `VehicleID`

The datareader for the VehicleID dataset.

### `CRAWLER_ARGS`

The crawler extracts all paths to tuples. 2 arguments:

- `data_folder`: path to the root VehicleID directory that contains the the `image` directory, e.g. `Data/VehicleID`
- `train_folder`: name of the `image` directory, e.g. `image`
- `attribute_folder`: The directory containing attributes. Use `attribute`.

### DATASET_ARGS

TorchDataset stes up functions to load images from path, and well as yield a single annotation/label. We need to provide the index of the annotation, since VehicleID has multiple annotations.
Each tuple of a sample is of the form: `(path/to/image, pid, cid, color, model)`
3 arguments:

- `pathidx`: The index where the path exists. This is `0` for VehicleID
- `annotationidx`: the index where the annotation exists. If you want to use the `color` annotation, use `3`. 
- `classificationclass`: The name of the annotation used. See tuple above. If you want to use the make annotation, use `color`

### GENERATOR_ARGS

Unused

### DATALOADER_ARGS

Unused

## `VeRi`

The datareader for the VeRi dataset.

### `CRAWLER_ARGS`

The crawler extracts all paths to tuples. 2 arguments:

- `data_folder`: path to the root VeRi directory that contains the the `image` directory, e.g. `data/VeRi`

### DATASET_ARGS

TorchDataset stes up functions to load images from path, and well as yield a single annotation/label. We need to provide the index of the annotation, since VeRi has multiple annotations.
Each tuple of a sample is of the form: `(path/to/image, pid, cid, color, type)`
3 arguments:

- `pathidx`: The index where the path exists. This is `0` for VehicleID
- `annotationidx`: the index where the annotation exists. If you want to use the `color` annotation, use `3`. 
- `classificationclass`: The name of the annotation used. See tuple above. If you want to use the make annotation, use `color`

### GENERATOR_ARGS

Unused

### DATALOADER_ARGS

Unused
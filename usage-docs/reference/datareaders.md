# DATAREADERS


DataReaders perform 3 tasks: 

1. They crawl a directory to collect paths to all files as well as associatated annotations
2. They build a TorchDataset for this data
3. They build a generator using TorchDataloader to yield batches

`EXECUTION.DATAREADERS` is a dictionary. It looks like:


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

`EXECUTION.DATAREADER.DATAREADER` determines which datareader we will use. In turn, this determines what arguments exist for `CRAWLER_ARGS`, `DATASET_ARGS`, `GENERATOR_ARGS`, and `DATALOADER_ARGS`.

The following values for `EXECUTION.DATAREADER.DATAREADER` are available:

- `CompCars`
- `VehicleColor`

Subsequent parameters for each `EXECUTION.DATAREADER.DATAREADER` is provided below.


## `CompCars`

The datareader for the CompCars dataset.

### `CRAWLER_ARGS`

The crawler extracts all paths to tuples. 2 arguments:

- `data_folder`: path to the root CompCars directory that contains the the `image` directory, e.g. `data/CompCars`
- `train_folder`: name of the `image` directory, e.g. `image`

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


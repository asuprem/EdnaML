Roadmap
- connector to Google Drive
   - and other cloud solutions
- sharded training with pre-generated features for text ( and images...?)
    - need a generic sharded base dataloader, maybe???
    - basically, a class that loads data into memory, except now it can be sharded...?
    - memory mapping, and formalize transformation and pre-sharding...
    - need connectors to data storage as ell. For now, we can just do file system, but eventually, s3, azureblob, etc
- need save details FIRST
    - when script with eml is run
    - during saving procedure
        - save the model, artifacts, AND the code file itself
        - code-file will append run-id (some random string, possibly based on UNIX time???) to the end
        - possibly integrate mlflow starting now????
            - mlflow tracking means in config, we pass in a few arguments in a metrics section
            - let's say accuracy...HOW????
            - ok, how about this -- each trainer implements its own metrics set, and based on trainer parameters, can calculate any and all metrics
            - custom metrics...how...
                - potentially like this
                - just like sklearn, to calculate metrics, we pass in the model output
                - so metrics must follow a metrics format
                - we need to create a BaseMetrics class, or alternatively, use TorchMetrics...
                - we will need to go through some metrics to get an idea for abstract class that wraps around torch metrics

- meta learning optimizations
    - mosaic/compose
- metrics tracking
    - 
- migrate the datasets in gcp, azure, aws
    - $50 per year
    - <  >
- live datasets
- config query engine
    - EdnaQuery???? or yaml query engine client and engine
    - engine indexes existing configs
    - for now, we won't "change" a config
    - when new configs are added to directorty, engine will index them as well
    - client used to search them
    - mongo database and client???

    - broad approach
        - EdnaQuery
        - point at folder or directory
        - monitor notifies when files have changed or been added. Send file into AddQueue
        - converter converts files into json
        - then indexer pushes into mongo
        - client connects to that mongo...ma
        - maybe written as EdnaJob???? with an EdnaStreamer managing this????

- data cleaning and management
    - EdnaData module
    - something like Dask wrapper?
    - do data statistics? and transformations?
    - realistic case:
        - first apply EdnaData (operates on the SAME configuration?????) to transform raw data (with class balancing, transformations, etc)
        - then EdnaML directly takes take and does not need to transform it at all...
        - similar to data provenance????
- excel sheet for roadmap
- post training hook; pre-training hook.
- document transformation has 3 components: TRANSFORMATION --> TRAIN_TRANSFORMAITON / TEST_TRANSFORMATION 🟢
- soft targets 🟢
- formalize crawler
- concept drift embeddings
    (within edna-data?)
- torchvision_datasets 🟡 
    - Have a torchvision datareader 🟢
        - Generator 🟢
    - complete all torchvision datareaders
    - integrate torchvision wrapper to fill dummy crawler, classes, and get_num_entities
- pytest
- batch size predictor
- cyclical learning rate guesser
- transformers
- GANs
- colabel-deploy
- weak supervision
- snorkel+eews
- fp16
- multi-gpu training
- kubernetes????
- azure?????
- BoxCars
- VehicleColors 🟢
- logging -- tensorboard???
- metrics logging
- apply()
    - also, when user wants to put custom things in Edna, they can use the custom add functions (addOptimizer, etc). This will do lazy setup
    - Then, when user does apply, this will apply the lazy changes internally.
        - custom Crawler    🟢
        - custom datareader 🟢
        - custom generator 🟢
        - custom model 🟢
        - custom loss 🟢
        - custom optimizer 🟢
        - custom scheduler 🟢
        - default datareader is just datareader (or MNIST) 
        - default Model is SimpleConvModel...
        
- better EdnaML functions -- internal vs conventional functions...

- Need to start thinking about config files and how to interact with them
    - overall scheme: each ednaml experiment needs a config file
    - when you run the file, it will save everything to a folder, and back it up if you want it to
    - in this folder, we currently have stored the model, training info, and log
    - we need to add a few things
    - a function in ednaml called eml.package() 
        - inside the folder we just specificed, it will create an eml <<<look up guild AI>>>
    - need to upgrade to azure, or in-house system, or cloudlab...


- EdnaML in test mode -- does not build train data loader...
--------------------------------------------------------------------
so, someone does the following??? ✅
EdnaML() <-- empty, so no configuration is set up> ✅
This means working from scratch ✅
So, at minimum, they need a datareader (with its crawler, generator, etc), a model, a loss ✅
eml.addCrawler(crawlerinstance) ✅





--------------------------------------------------------------------
How to improve apply() function? ✅
First step in apply is verification of current configuration. What does that look like? ✅
addDataloader() ✅
addModel()/updateModel() ✅
addOptimizer/updateOptimizer() ✅
addScheduler()/updateScheduler() ✅
addLossArray()/updateLossArray() ✅
addLossOptimizer()/update ✅
add/update LossScheduler ✅


--------------------------------------------------------------------
soft_targets: true
soft_target_branch: ['vcolor', 'vtype']
soft_target_output_source: fuse

if soft targets true, model internally stores this value as well as the name of the output source. Then, secondary outputs gets a list of these soft target `guesses`
--------------------------------------------------------------------








python3 -m build
python3 -m twine upload --repository pypi dist/*


Next steps
--------------------
type hinting
convert local drve_backup to cfg.drive_backup in EdnaML ✅
Create context manager
Fix configs to reflect OPTIMIZER_NAME -> OPTIMIZER 

Make Optimizer into list of optimizers in config    ✅
    Each optimizer has a name   ✅
    if No name, assign a name   ✅
    For Classification/MultiClassification/MultiBranchTrainer --> they take the first optimizer from the list. GANTrainer will look for specific optimizers, etc ✅
Make Scheduler into list of schedulers  ✅
    Each scheduler has a name tied to optimizer ✅


Changelog
    CoLabelTrainer --> ClassificationTrainer ✅

multibranchresnet --> load_param ✅
MultiBranchResnet --> Everything ✅




utils.blocks propagate  ✅
test new multiclassification that has updated softmax dimensions ✅
add soft targets for loss as well


model_output_count, model_feature_count in modelabstract ✅

need to create an OutputMetadata class, a BranchMetadata class
    OutputMetadata contains metadata for a single output, such as name, dimensions, and label it is tracking
    BranchMetadata tracks a single branch, and what outputs it has through a list of OutputMetadatas
Need to fix BoseOptimizer to return BaseOptimizer, then in trainer, reference optimizer.optimizer, instead of just optimizer
Need them stored as dicts in trainer, keyed by their name
Same for scheduler, loss scheduler, etc...


Need to fix or adjust the loss and outputs to deal with re-id losses.
    make a taxonomy of the types of outputs from a model...

Need to unify where we are looking for loss_name, model output; need unified design doc for this as well.


make a verifier, that checks some naming conventions
    if it is multiclassification, make sure that loss name matches one of the classificationclasses...


have a remove_softmax option to remove the softmax dimensions and use only features...potentially for reid

Fix SOFTMAX_DIM in colbl_vcolor yamls
change soft_dimensions to softmax_dimensions ✅

Fix TRAINDATA as dict all the time, then propagate this through...✅

Rename CoLabelResNet to ClassificationResNet ✅


------------------------------------------------------------------------------------------------------------------------------------

LabelIntegrator


LabelFunction : a label function base class
    - open label function/model
        - given some data, returns (label, trust score)
    - closed label function/model
        - given some data, returns a label


LabelIntegrators : an integrator that takes in a few noisy labels + metadata and yields a combined label
    - PGMIntegrator: takes in a set of labels, and yields a combined label
    - EEWSIntegrator: takes in a set of labels, plus the raw data, plus a downsteam-configuration, and yields a cmbined label+downstream model
    - TrustIntegrator: takes in a set of (labels, TrustScores), 

















batch size predictor

learning rate guesser
    also, cyclical learning rates, etc

membatch method where the batch update is every few batches...?
    accumulation_steps = 10
    for i, batch in enumerate(batches):
        # Scale the loss to the mean of the accumulated batch size
        loss = calculate_loss(batch) / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            # Reset gradients, for the next accumulated batches
            optimizer.zero_grad()

Logging...


EdnaPipeline, for composing multiple models...?
SimpleEdnaML???


To test:

    
    then we can move to the multiclassification stuff..






Crawler Types:
    
    zsl/gzsl crawlers
        TODO
    re-id crawlers
        metadata -> train/test/val
            crawl->
    classification crawlers
        reads through data and yields
            metadata->train/test/val
                crawl->[(annot,annot,annot,annot,path),...]
                class->{annot:num-classes, annot:num-classes, annot:num-classes}
            classes:
                annot:{id:nam,id:name,id:name}
                annot:{id:nam,id:name,id:name}
        
Generator Types:
    
    zsl/gzsl generator
    
    re-id generator

    classification generator
        take a crawler, and train test splits
        set up a dataset...

        single-annotation generator

        multi-annotation generator



What is customization:
    crawler for reading a dataset
    torchdataset for converting from tuple to dataset
    collatefn to convert from torchdataset to tensor in torchdataloader

config
    dataloader -> Cars196Dataloader
        crawler-args:<>
            train-folder
            test-folder
            
        dataset-args:<>
        dataoader-args:<>



load_param <-- needs to be fixed w.r.t. multiple branches...

    <-- called by CoLabelInterpretableResNet <-- build_base



Note -- CoLabelGenerator is only for VehicleColor?????

NOTE: updated the log backup restore in colabel.py


CoLabelIntegratedDatasetCrawler
CoLabelIntegratedDatasetGenerator



CoDataset
    -> image
    -> readable
        -> make.txt
        -> type.txt
        -> color.txt
    -> splits
        -> vmmr
            -> train.txt  <type>/<color>/[<makeid>/<modelid>/3ac218c0c6c378.jpg]
            -> test.txt








for batch in dataloader:
    all_input_ids, <all_attention_mask>, all_token_type_ids, <all_masklm>, all_labels, <all_datalabels> = batch
    encoded = midas.encoder(batch)
    # calcualte epsilon...

    get the training datas for the experts. For each expert
        pass its training data through midas encoder-decoder
        Then through expert.
        Get training data point cluster center in embedding space.
        Get all points near this training data point (let's say 20)
        Get these data points in the midas embedding space
        Get distances, to cluster center of 20 points, then calculate max(dist-features / dist-embedding)

    [perturbed]*numpoints = perturb(encoded, numpoints, epsilon)

    [theta]*numpoints = get_distance(perturbed, encoded)
    
    for decoder in midas
        [decoded]*numdomains*numpoints= midas.decoder(perturbed)
    
    for expert:
        predict, logits = expert(predict)

        theta_e = get_distance(logits_perturbed, logits_original)

        L_e = theta_e / theta
        max(l_e) is the L_e for this expert

    then take the min(l_e)
    that expert's results are the results...

    get these values
    compare to true ground truth... 
    
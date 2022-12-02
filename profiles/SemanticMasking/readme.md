# Experiments for Semantic Masking

## Experimental Setup

We use 11 fake news datasets. On each dataset, we perform 7 experiments:

1. nomask
2. rtm
3. rwm
4. tfidf-tm
5. tfidf-wm
6. att-wm
7. att-tm

## How to run experiments

Experiments are created by composing basic blocks through EdnaML configuration files. At a high level, we can control the type of experiment with the following cell from []:


```
# Select the dataset to perform masking experiment on
# efnd | fakeddit | nela2018 | nela2019 | nela | nela-elections | nela-gt | nela2021  # The latter 3 are for 2020
dataset = "efnd"    

# Select sub-dataset, if applicable. NOTE: This is ONLY if `dataset=efnd`. For any other value of dataset, this can be empty string, or any string. 
# cmu_miscov19 | kagglefn_short | kagglefn_long | cov19_fn_title | cov19_fn_text | coaid_news | cov_rumor | covid_fn | covid_cq
subdataset = "cov19_fn_text"

# Select the masking experiment.
# nomask | rtm | rwm | ktm_tfidf | kwm_tfidf | ktm_att | kwm_att | ktrtm_tfidf | kwrwm_tfidf | ktrtm_att | kwrwm_att
masking = "nomask"  

# Select the base model. This should correspond to a HuggingFace name, e.g. albert-base-v2
model = "albert-base-v2"

# Additional parameters
backup = False          # Whether to back up to cloud storage
epochs = 2              # How many epochs to train with
save_frequency = 1      # How often to save intermediate models

# Extra parameters for KMeansProxy and Lipschitz Scores. Unneeded here.
k_alpha = 0.1 # (0, 1)
l_alpha = 0.1 # (0, 1)
```

## `Dataset` and `Subdataset`

The choice of `dataset` and `subdataset` adjusts the arguments provided to the Crawler, since each dataset has slightly different arguments needed to crawl it into memory.

The snippet below shows what happens if `dataset` is `efnd` or `nela` (the latter refers to the `nela2020` dataset):

```
if dataset == "efnd":  
  crawler_file = "./EdnaML/profiles/FNC/experts/crawlers/efnd-crawler.py"
  crawler_variant = "./EdnaML/profiles/FNC/experts/configs/crawler-efnd-base.yml"
  crawler_args = {
      "data_folder" : "Data",
      "include": [subdataset]
  }
  model_qualifier = subdataset
elif dataset == "nela": #covid2020   
  crawler_file = "./EdnaML/profiles/FNC/experts/crawlers/nela-crawler.py"
  crawler_variant = "./EdnaML/profiles/FNC/experts/configs/crawler-nela-base.yml"
  crawler_args = {
      "data_folder" : "Data",
      "sub_folder" : "nela-covid-2020"
  }
  model_qualifier = "nela_covid_2020"
...
```

## `masking`

Masking selection adjusts the 

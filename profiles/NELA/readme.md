# Executing an Edna workflow


## EdnaML

This is for an ML training and testing pipeline. Runs from an EdnaML config, e.g. nela-covid-v0.yml.

Current:
    Have custom classes in colab cell
    Set up an ednaml class
    add custom classes
    apply()
    train()
    eval()

Future Roadmap for this:
    have yml file with options
    have main.py or run.py or classes.py file with custom classes WITH edna decorators
    then execute> edna-ml classes.py config.yml --options
        ednaml will create an EdnaML(options) classes, then add requisite custom classes using decorators, then do train() and eval()


## EdnaDeploy

This is after an ML model is trained and tested. We use it to predict things. Runs from a deployment confit, e.g. nela-covid-deploy.yml

Current:
    Have same custom classes in colab cell (can extend existing colab notebook for deploy)
    Set up an EdnaDeploy class: 
    Add custom classes (incl deploy class and custom data crawler class for prediction data...)
    apply()
    deploy()

Next step future roadmap for this:
    have yml file with options (nela-covid-deploy.yml)
    have original yml file from ednaml (config.yml)
    have a classes.py with custom classes WITH edna decorators
    then execute>   edna-deploy classes.py config.yml nela-covid-deploy.yml --options
        edna-deploy will merge config.yml and nela-covid-deploy.yml (the latter contains deployment details for deployment and deployment crawler)
        then add the custom classes using decorators (i.e. the custom model, crawler, ets. BUT there is a hierarchy between the deploymentCrawler and the ednamlcrawler)
        Then do ed.deploy()


Future Roadmap for this
    We have a valid config stored in mongo, along with the associated best run (that has a v0, etc)
    we can make an edna-query to find the best run using some run-id (SCHEMA development!!!!)
    then edna-deploy run-id deployment.yml classes.py
        run-id will fetch the associated classes.py from that run-id, specifically for our model
        Then set up the classes, and get the original config.
        Merge that config with deployment.yml
        Then load replacement classes from classes.py, specifically deployment classes (i.e. deploymentcrawler and EdnaDeploy)
        Finally, check if deployment.yml already has a copy in mongo (for now, no checking because we ain;t backing up anything)
        then set up edna-deploy, and ed.deploy()

## EdnaServe

This is a serve job, so deal with this later

kubernetes deployment / 
log in azure, start a longterm vm
pytorch-serve <pytorch-model.pth>



## EdnaJob

Deal with this later

<ingest> download from twitter
<process> https://model-ip/v1/get-fake-news?text="twitter-text"
<emit> table.insert(text, false/true)

## EdnaData

This is an EdnaJob, so deal with this later

<ingest> from some live source
<process>
<emit> -> azure, gdrive, etc




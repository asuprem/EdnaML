import ednaml, torch, os, csv
from ednaml.crawlers import Crawler
from ednaml.deploy.BaseDeploy import BaseDeploy





class NELADeploy(BaseDeploy):


    def deploy_step(self, batch):
        return super().deploy_step(batch)

    def output_setup(self, **kwargs):
        return super().output_setup(**kwargs)

    def output_step(self, logits, features, secondary):
        return super().output_step(logits, features, secondary)




import click
@click.argument("config")
@click.argument("mode")
def main(config, mode):
    from ednaml.core import EdnaDeploy
    import main
    eml = EdnaDeploy(config=config)
    eml.addCrawlerClass(main.NELACrawler)
    eml.addModelClass(main.NELAModel)
    eml.addDeploymentClass(NELADeploy)

    eml.apply(input_size=(eml.cfg.TRAIN_TRANSFORMATION.BATCH_SIZE,eml.cfg.EXECUTION.DATAREADER.DATASET_ARGS["maxlen"]),
          dtypes=[torch.long])

    eml.train()

if __name__ == "__main__":
    main()
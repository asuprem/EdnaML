from core.EdnaML import EdnaML




def main(config, mode, weights):

    ednaml = EdnaML(config, "train")
    ednaml.addConfig(config)
    #ednaml.autoSetup()
    ednaml.setup()
    ednaml.buildDataloader()
    ednaml.buildModel()
    ednaml.downloadWeights()
    ednaml.loadWeights()
    ednaml.buildLoss()
    ednaml.buildOptimizer()
    ednaml.buildScheduler()
    ednaml.checkPreviousStop()
    ednaml.buildTrainer()



    ednaml.train()  # includes buildTrainer()
    ednaml.evaluate()



    --

    ednaml = EdnaML(config, mode, weights)
    EdnaML.quicksetup()
    
    EdnaML.train()
    EdnaML.evaluate()
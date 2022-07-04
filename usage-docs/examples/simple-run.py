from ednaml.core import EdnaML



if __name__=="__main__":
    eml = EdnaML("../sample-configs/datasets/config-veri.yml",  "train")

    eml.quickSetup()
    
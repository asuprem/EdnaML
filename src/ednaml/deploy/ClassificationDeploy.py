
import torch
from ednaml.deploy.BaseDeploy import BaseDeploy

class ClassificationDeploy(BaseDeploy):
    def deploy_step(self, batch): 
        (
            img,
            labels,
        ) = batch  
        # logits, features, labels
        logits, features, secondary = self.model(img)
        
        return logits, features, (secondary, labels)

    def output_setup(self, **kwargs):
        self.total_pred = []
        self.total_labels = []
        try:
            from sklearn.metrics import classification_report as creport
            self.creport = creport
        except ImportError:
            self.creport = self.accprint
    def output_step(self, logits, features, secondary):
        self.total_pred += torch.argmax(logits, dim=1).tolist()
        self.total_labels += secondary[-1].tolist()

    def end_of_epoch(self, epoch: int):
        print(self.creport(self.total_labels, self.total_pred))

    def creport(self, labels, pred):
        return "Accuracy:  %f"%((torch.tensor(self.total_pred) == torch.tensor(self.total_labels)).sum().float() / float(
            len(self.total_labels)
        ))

    


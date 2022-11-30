



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
        pass
    def output_step(self, logits, features, secondary):
        import pdb
        pdb.set_trace()
from . import SimpleTrainer
import torch, tqdm

class VehicleIDTrainer(SimpleTrainer):
    try:
        #apex = __import__('apex')
        # TODO TODO TODO HIGH PRIORITY APEX disabled because of loss optimizer. Do not know if apex will work without scaled_loss on loss_optimizer's model, which is actually a "list" of models...
        apex = None
    except:
        apex = None
    def evaluate(self):
        self.model.eval()
        features, pids, cids = [], [], []
        with torch.no_grad():
            for batch in tqdm.tqdm(self.test_loader, total=len(self.test_loader), leave=False):
                data, pid, camid, img = batch
                data = data.cuda()
                feature = self.model(data).detach().cpu()
                features.append(feature)
                pids.append(pid)
                cids.append(camid)
        features, pids, cids = torch.cat(features, dim=0), torch.cat(pids, dim=0), torch.cat(cids, dim=0)

        query_features, gallery_features = features[:self.queries], features[self.queries:]
        query_pid, gallery_pid = pids[:self.queries], pids[self.queries:]
        query_cid, gallery_cid = cids[:self.queries], cids[self.queries:]
        
        distmat = self.cosine_query_to_gallery_distances(query_features, gallery_features)
        #distmat=  distmat.numpy()
        self.logger.info('Validation in progress')
        v_cmc, v_mAP = self.eval_vid(distmat, query_pid.numpy(), gallery_pid.numpy(), query_cid.numpy(), gallery_cid.numpy(), 100)
        self.logger.info('Completed VehicleID CMC')

        self.logger.info('Completed mAP Calculation')
        self.logger.info('VeRi_mAP: {:.2%}'.format(v_mAP))
        for r in [1,2, 3, 4, 5,10,15,20]:
            self.logger.info('VeRi CMC Rank-{}: {:.2%}'.format(r, v_cmc[r-1]))
  
    #def __evaluate(self):
    #    pass
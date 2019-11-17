import tqdm
from collections import defaultdict, OrderedDict
from sklearn.metrics import average_precision_score
import shutil
import os
import torch
import numpy as np
from scipy.spatial.distance import cdist
import loss.builders

from .BaseTrainer import BaseTrainer

import pdb

class SimpleTrainer(BaseTrainer):
    try:
        #apex = __import__('apex')
        # TODO TODO TODO HIGH PRIORITY APEX disabled because of loss optimizer. Do not know if apex will work without scaled_loss on loss_optimizer's model, which is actually a "list" of models...
        apex = None
    except:
        apex = None
    def __init__(   self, 
                    model: torch.nn.Module, 
                    loss_fn: loss.builders.LossBuilder, 
                    optimizer: torch.optim.Optimizer, loss_optimizer: torch.optim.Optimizer, 
                    scheduler: torch.optim.lr_scheduler._LRScheduler, loss_scheduler: torch.optim.lr_scheduler._LRScheduler, 
                    train_loader, test_loader, 
                    queries: int, epochs: int, logger, **kwargs):   #kwargs includes crawler
        
        super(SimpleTrainer,self).__init__(model, loss_fn, optimizer, loss_optimizer, scheduler, loss_scheduler, train_loader, test_loader, epochs, logger)
        
        self.queries = queries
        self.loss = []
        self.softaccuracy = []

        self.crawler = kwargs.get("crawler", None)

    # setup inherited from BaseTrainer
    def step(self,batch):
        self.model.train()
        self.optimizer.zero_grad()
        if self.loss_optimizer is not None: # In case loss object doesn;t have any parameters, this will be None. See optimizers.StandardLossOptimizer
            self.loss_optimizer.zero_grad()
        batch_kwargs = {}
        img, batch_kwargs["labels"] = batch
        img, batch_kwargs["labels"] = img.cuda(), batch_kwargs["labels"].cuda()
        # logits, features, labels
        batch_kwargs["logits"], batch_kwargs["features"] = self.model(img)
        batch_kwargs["epoch"] = self.global_epoch   # For CompactContrastiveLoss
        loss = self.loss_fn(**batch_kwargs)
        if self.fp16 and self.apex is not None:
            with self.apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()
        if self.loss_optimizer is not None: # In case loss object doesn;t have any parameters, this will be None. See optimizers.StandardLossOptimizer
            self.loss_optimizer.step()
        
        self.loss.append(loss.cpu().item())
        
        if batch_kwargs["logits"] is not None:
            softmax_accuracy = (batch_kwargs["logits"].max(1)[1] == batch_kwargs["labels"]).float().mean()
            self.softaccuracy.append(softmax_accuracy.cpu().item())
        else:
            self.softaccuracy.append(0)
        
        

    def train(self,continue_epoch = 0):    
        self.logger.info("Starting training")
        self.logger.info("Logging to:\t%s"%self.logger_file)
        self.logger.info("Models will be saved to local directory:\t%s"%self.save_directory)
        if self.save_backup:
            self.logger.info("Models will be backed up to drive directory:\t%s"%self.backup_directory)
        self.logger.info("Models will be saved with base name:\t%s_epoch[].pth"%self.model_save_name)
        self.logger.info("Optimizers will be saved with base name:\t%s_epoch[]_optimizer.pth"%self.model_save_name)
        self.logger.info("Schedulers will be saved with base name:\t%s_epoch[]_scheduler.pth"%self.model_save_name)
        

        if continue_epoch > 0:
            load_epoch = continue_epoch - 1
            self.load(load_epoch)

        self.logger.info("Performing initial evaluation...")
        self.initial_evaluate()

        for epoch in range(self.epochs):
            if epoch >= continue_epoch:
                for batch in self.train_loader:
                    if not self.global_batch:
                        lrs = self.scheduler.get_lr(); lrs = sum(lrs)/float(len(lrs))
                        self.logger.info("Starting epoch {0} with {1} steps and learning rate {2:2.5E}".format(epoch, len(self.train_loader) - (len(self.train_loader)%10), lrs))
                    self.step(batch)
                    self.global_batch += 1
                    if (self.global_batch + 1) % self.step_verbose == 0:
                        loss_avg = sum(self.loss[-100:]) / float(len(self.loss[-100:]))
                        soft_avg = sum(self.softaccuracy[-100:]) / float(len(self.softaccuracy[-100:]))
                        self.logger.info('Epoch{0}.{1}\tTotal Loss: {2:.3f} Softmax: {3:.3f}'.format(self.global_epoch, self.global_batch, loss_avg, soft_avg))
                self.global_batch = 0
                self.scheduler.step()
                if self.loss_scheduler is not None:
                    self.loss_scheduler.step()
                self.logger.info('{0} Completed epoch {1} {2}'.format('*'*10, self.global_epoch, '*'*10))
                if self.global_epoch % self.test_frequency == 0:
                    self.evaluate()
                if self.global_epoch % self.save_frequency == 0:
                    self.save()
                self.global_epoch += 1
            else:
                self.global_epoch = epoch+1


    # https://github.com/Jakel21/vehicle-ReID-baseline/blob/master/vehiclereid/eval_metrics.py
    def eval_vid(self,distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
        """Evaluation with veri metric
        """
        num_q, num_g = distmat.shape

        if num_g < max_rank:
            max_rank = num_g
            print('Note: number of gallery samples is quite small, got {}'.format(num_g))

        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        # compute cmc curve for each query
        all_cmc = []
        all_AP = []
        num_valid_q = 0.  # number of valid query
    
        for q_idx in range(num_q):
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            #remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            remove = (g_pids[order] == -1)
            keep = np.invert(remove)

            # compute cmc curve
            raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
            if not np.any(raw_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = raw_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.

            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = raw_cmc.sum()
            tmp_cmc = raw_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        return all_cmc, mAP


    def evaluate(self):
        self.model.eval()
        features, pids, cids, imgs = [], [], [], []
        with torch.no_grad():
            for batch in tqdm.tqdm(self.test_loader, total=len(self.test_loader), leave=False):
                data, pid, camid, img = batch
                data = data.cuda()
                feature = self.model(data).detach().cpu()
                features.append(feature)
                pids.append(pid)
                cids.append(camid)
                imgs+=list(img)
        
        # For market 1501
        features, pids, cids = torch.cat(features, dim=0), torch.cat(pids, dim=0), torch.cat(cids, dim=0)
        
        if self.crawler is not None:
            track_features = [torch.zeros(features[0].shape)]*len(self.crawler.metadata["track"]["crawl"])
            track_pids = [0]*len(self.crawler.metadata["track"]["crawl"])
            track_cids = [0]*len(self.crawler.metadata["track"]["crawl"])
            track_count = [0]*len(self.crawler.metadata["track"]["crawl"])
            pdb.set_trace()
            for feature, img in zip(features[self.queries:], imgs[self.queries:]):  # use only gallery features, no query features
                track_idx = self.crawler.metadata["track"]["dict"][img]
                track_count[track_idx]+=1
                track_pid = self.crawler.metadata["track"]["info"][track_idx]["pid"]
                track_cid = self.crawler.metadata["track"]["info"][track_idx]["cid"]
                track_features[track_idx] += feature
                track_pids[track_idx] = track_pid
                track_cids[track_idx] = track_cid
            for idx,feats in enumerate(track_features): # Get average of features
                track_features[idx] = feats / track_count[idx]  # maybe max...?
            
            pdb.set_trace()
            track_features, track_pids, track_cids = torch.cat(track_features, dim=0), torch.cat(track_pids, dim=0), torch.cat(track_cids, dim=0)
        else:
            track_features, track_pids, track_cids = None, None, None
        
        
        

        query_features, gallery_features = features[:self.queries], features[self.queries:]
        query_pid, gallery_pid = pids[:self.queries], pids[self.queries:]
        query_cid, gallery_cid = cids[:self.queries], cids[self.queries:]
        
        #distmat = self.cosine_query_to_gallery_distances(query_features, gallery_features)
        distmat = self.query_to_gallery_distances(query_features, gallery_features)
        if track_features is not None:
            track_distmat = self.query_to_gallery_distances(query_features, track_features)
        #distmat=  distmat.numpy()
        self.logger.info('Validation in progress')
        #m_cmc, mAP, _ = self.eval_func(distmat, query_pid.numpy(), gallery_pid.numpy(), query_cid.numpy(), gallery_cid.numpy(), 50)
        m_cmc = self.cmc(distmat, query_ids=query_pid.numpy(), gallery_ids=gallery_pid.numpy(), query_cams=query_cid.numpy(), gallery_cams=gallery_cid.numpy(), topk=100, separate_camera_set=False, single_gallery_shot=False, first_match_break=True)
        self.logger.info('Completed market-1501 CMC')
        c_cmc = self.cmc(distmat, query_ids=query_pid.numpy(), gallery_ids=gallery_pid.numpy(), query_cams=query_cid.numpy(), gallery_cams=gallery_cid.numpy(), topk=100, separate_camera_set=True, single_gallery_shot=True, first_match_break=False)
        self.logger.info('Completed CUHK CMC')
        if track_features is not None:
            v_cmc = self.cmc(track_distmat, query_ids=query_pid.numpy(), gallery_ids=track_pids.numpy(), query_cams=query_cid.numpy(), gallery_cams=track_cids.numpy(), topk=100, separate_camera_set=False, single_gallery_shot=False, first_match_break=True)
            v_mAP = self.mean_ap(track_distmat, query_ids=query_pid.numpy(), gallery_ids=track_pids.numpy(), query_cams=query_cid.numpy(), gallery_cams=track_cids.numpy())
            self.logger.info('Completed VeRi-776 CMC')
        
        mAP = self.mean_ap(distmat, query_ids=query_pid.numpy(), gallery_ids=gallery_pid.numpy(), query_cams=query_cid.numpy(), gallery_cams=gallery_cid.numpy())


        self.logger.info('Completed mAP Calculation')
        
        if track_features is not None:
            for r in [1,2, 3, 4, 5,10,15,20]:
                self.logger.info('VeRi CMC Rank-{}: {:.2%}'.format(r, v_cmc[r-1]))
            self.logger.info('VeRi-mAP: {:.2%}'.format(v_mAP))
        self.logger.info('mAP: {:.2%}'.format(mAP))
        #self.logger.info('VID_mAP: {:.2%}'.format(v_mAP))
        for r in [1,2, 3, 4, 5,10,15,20]:
            self.logger.info('Market-1501 CMC Rank-{}: {:.2%}'.format(r, m_cmc[r-1]))
        for r in [1,2, 3, 4, 5,10,15,20]:
            self.logger.info('CUHK CMC Rank-{}: {:.2%}'.format(r, c_cmc[r-1]))
        
  
    def query_to_gallery_distances(self, qf, gf):
        # distancesis sqrt(sum((a-b)^2))
        # so a^2 + b^2 - 2ab
        a2b2 = torch.pow(qf, 2).sum(1, keepdim=True).expand(qf.size(0), gf.size(0))
        a2b2 = a2b2 + torch.pow(gf, 2).sum(1, keepdim=True).expand(gf.size(0), qf.size(0)).t()
        eu= torch.addmm(1,a2b2, -2, qf, gf.t())
        eu = eu.clamp(min=1e-12).sqrt()
        return eu

    def cosine_query_to_gallery_distances(self, qf, gf):
        # distancesis sqrt(sum((a-b)^2))
        # so a^2 + b^2 - 2ab
        return cdist(qf, gf, metric='cosine')



    # https://github.com/Cysu/open-reid/blob/master/reid/evaluation_metrics/ranking.py
    def mean_ap(self,distmat, query_ids=None, gallery_ids=None,
            query_cams=None, gallery_cams=None):
        distmat = distmat
        m, n = distmat.shape
        if query_ids is None:
            query_ids = np.arange(m)
        if gallery_ids is None:
            gallery_ids = np.arange(n)
        if query_cams is None:
            query_cams = np.zeros(m).astype(np.int32)
        if gallery_cams is None:
            gallery_cams = np.ones(n).astype(np.int32)
        query_ids = np.asarray(query_ids)
        gallery_ids = np.asarray(gallery_ids)
        query_cams = np.asarray(query_cams)
        gallery_cams = np.asarray(gallery_cams)
        # Sort and find correct matches
        indices = np.argsort(distmat, axis=1)
        matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
        # Compute AP for each query
        aps = []
        for i in range(m):
            # Filter out the same id and same camera
            valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                    (gallery_cams[indices[i]] != query_cams[i]))
            y_true = matches[i, valid]
            y_score = -distmat[i][indices[i]][valid]
            if not np.any(y_true): continue
            aps.append(average_precision_score(y_true, y_score))
        if len(aps) == 0:
            raise RuntimeError("No valid query")
        return np.mean(aps)

    # https://github.com/Cysu/open-reid/blob/master/reid/evaluation_metrics/ranking.py
    def cmc(self,distmat, query_ids=None, gallery_ids=None,
        query_cams=None, gallery_cams=None, topk=100,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=False):
        m, n = distmat.shape
        # Sort and find correct matches
        indices = np.argsort(distmat, axis=1)
        matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
        # Compute CMC for each query
        ret = np.zeros(topk)
        num_valid_queries = 0
        for i in range(m):
            # Filter out the same id and same camera
            valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                    (gallery_cams[indices[i]] != query_cams[i]))
            if separate_camera_set:
                # Filter out samples from same camera
                valid &= (gallery_cams[indices[i]] != query_cams[i])
            if not np.any(matches[i, valid]): continue
            if single_gallery_shot:
                repeat = 10
                gids = gallery_ids[indices[i][valid]]
                inds = np.where(valid)[0]
                ids_dict = defaultdict(list)
                for j, x in zip(inds, gids):
                    ids_dict[x].append(j)
            else:
                repeat = 1
            for _ in range(repeat):
                if single_gallery_shot:
                    # Randomly choose one instance for each id
                    sampled = (valid & self._unique_sample(ids_dict, len(valid)))
                    index = np.nonzero(matches[i, sampled])[0]
                else:
                    index = np.nonzero(matches[i, valid])[0]
                delta = 1. / (len(index) * repeat)
                for j, k in enumerate(index):
                    if k - j >= topk: break
                    if first_match_break:
                        ret[k - j] += 1
                        break
                    ret[k - j] += delta
            num_valid_queries += 1
        if num_valid_queries == 0:
            raise RuntimeError("No valid query")
        return ret.cumsum() / num_valid_queries

    def _unique_sample(self,ids_dict, num):
        mask = np.zeros(num, dtype=np.bool)
        for _, indices in ids_dict.items():
            i = np.random.choice(indices)
            mask[i] = True
        return mask


    def eval_func(self,distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        """Evaluation with market1501 metric
            Key: for each query identity, its gallery images from the same camera view are discarded.
            """
        num_q, num_g = distmat.shape
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        # compute cmc curve for each query
        all_cmc = []
        all_AP = []
        num_valid_q = 0.  # number of valid query
        for q_idx in range(num_q):
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)

            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            orig_cmc = matches[q_idx][keep]
            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.

            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        return all_cmc, mAP, all_AP

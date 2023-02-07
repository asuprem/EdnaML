# Implementation of SoftTriple Loss
import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
from ednaml.loss import Loss

# https://github.com/idstcv/SoftTriple/blob/master/loss/SoftTriple.py
class SoftTriple(Loss):
    def __init__(self, **kwargs):
        super(SoftTriple, self).__init__()
        self.la = kwargs.get("la")
        self.gamma = 1./kwargs.get("gamma")
        self.tau = kwargs.get("tau")
        self.margin = kwargs.get("margin")
        self.cN = kwargs.get("cN")
        self.K = kwargs.get("K")
        self.fc = Parameter(torch.Tensor("dim", self.cN*self.K))
        self.weight = torch.zeros(self.cN*self.K, self.cN*self.K, dtype=torch.bool).cuda()
        for i in range(0, self.cN):
            for j in range(0, self.K):
                self.weight[i*self.K+j, i*self.K+j+1:(i+1)*self.K] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return

    def forward(self, input, target):
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc*self.gamma, dim=2)
        simClass = torch.sum(prob*simStruc, dim=2)
        marginM = torch.zeros(simClass.shape).cuda()
        marginM[torch.arange(0, marginM.shape[0]), target] = self.margin
        lossClassify = F.cross_entropy(self.la*(simClass-marginM), target)
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/(self.cN*self.K*(self.K-1.))
            return lossClassify+self.tau*reg
        else:
            return lossClassify
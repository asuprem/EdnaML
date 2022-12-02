import math, torch
from torch import nn


class KMeansProxy(nn.Module):
    def __init__(self, proxy_factor, num_classes, embedding_dimensions):
        super().__init__()
        self.proxy_factor = proxy_factor
        self.num_classes = num_classes
        self.clusters = self.proxy_factor * self.num_classes
        self.embedding_dimensions = embedding_dimensions
        self.lr = 1e-2
        self.proxies = nn.Parameter(
            torch.zeros(self.clusters, self.embedding_dimensions)
        )
        self.labels = torch.zeros(self.clusters, self.num_classes)

        self._proxiesFlag = False
        self._optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.cost_window = 25
        self.costs = []
        self.loss = nn.MSELoss()

    def _initialize(self, x):

        samples_per_cluster = math.ceil(float(x.size(0))) / self.clusters
        assignment = list(range(0, self.clusters)) * int(samples_per_cluster)
        length_difference = len(assignment) - x.size(0)
        if length_difference < 0:
            assignment = assignment + assignment[length_difference:]
        else:
            assignment = assignment[: len(assignment) - length_difference]
        randomized_order = torch.randperm(len(assignment))
        randomized_assignment = torch.LongTensor(assignment)[randomized_order]

        batch_dim = 0

        for i in range(self.clusters):
            self.proxies.data[i] += x[randomized_assignment == i].mean(batch_dim)

        return randomized_assignment

    def _assign(self, x):
        return ((x[:, None] - self.proxies) ** 2).mean(2).argmin(1)

    def forward(self, x):
        if self.training:
            self.fit(x)
        return x, self.proxies[self._assign(x)], self.labels[self._assign(x)]

    def transform(self, x):
        return self._assign(x)

    def _update(self, x, cluster):
        self._optimizer.zero_grad()
        means = self.proxies[cluster]
        cur_cost = self.loss(x, means)
        cur_cost.backward()
        self._optimizer.step()
        return cur_cost.item()

    def fit(self, x):
        if not self._proxiesFlag:
            self._initialize(x)
            self._proxiesFlag = True
        clusters = self._assign(x)
        cur_cost = self._update(x, clusters)
        # costs.append(cur_cost)

    def labelUpdate(self, model, return_tuple_index=None):
        for idx, x in enumerate(self.proxies):
            return_value = model(x)
            self.labels[idx, :] = return_value[return_tuple_index]

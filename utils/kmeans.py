import random

import torch


class KMeans:
    def __init__(self, n_clusters=6, max_iter=None, verbose=False, device=torch.device("cpu")):
        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None  # shape: [x.shape[0],n_cluster]
        self.centers = None
        self.variation = torch.Tensor([float("Inf")]).to(device)
        self.verbose = verbose
        self.representative_samples = None
        self.max_iter = max_iter
        self.count = 0
        self.device = device

    def _k_init(self, X):
        self.started = False
        n_samples, n_features = X.shape
        centers = torch.zeros((self.n_clusters, n_features)).to(self.device)
        center_id = random.randint(0, n_samples-1)
        centers[0] = X[center_id]
        distances = torch.zeros(n_samples).to(self.device)
        for c in range(1, self.n_clusters):
            distances += torch.cdist(centers[(c-1):c], X).squeeze(0)
            prob = distances / torch.sum(distances)
            prob_threshold = torch.cumsum(prob, dim=0)
            best_candidate = (prob_threshold > random.random()).sum().clamp(max=n_samples-1)
            centers[c] = X[best_candidate]

        self.centers = centers
    
    def fit(self, X):
        self._k_init(X)
        while True:
            # 聚类标记
            self.nearest_center(X)
            # 更新中心点
            self.update_center(X)
            if self.verbose:
                print(self.variation, torch.argmin(self.dists, dim=0))
                
            if torch.abs(self.variation) < 1e-3 and self.max_iter is None:
                break
            elif self.max_iter is not None and self.count == self.max_iter:
                break

            if torch.isnan(self.variation):
                self._k_init(X)
            
            self.count += 1

        self.representative_samples = torch.argmin(self.dists, dim=1)

    def nearest_center(self, X):
        dists = torch.cdist(self.centers, X)
        self.labels = torch.argmin(dists, dim=1)
        if self.started:
            self.variation = torch.sum(self.dists - dists)
            
        self.dists = dists
        self.started = True

    def update_center(self, X):
        centers = torch.empty((0, X.shape[1])).to(self.device)
        for i in range(self.n_clusters):
            mask = torch.where(self.labels == i)
            cluster_samples = X[mask]
            centers = torch.cat((centers, torch.mean(cluster_samples, dim=0).unsqueeze(0)), dim=0)

        self.centers = centers

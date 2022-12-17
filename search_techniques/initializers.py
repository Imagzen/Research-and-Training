import numpy as np
from sklearn.cluster import KMeans
from config import *
import os

class KMeansSearchInitializer:

    def calculate_clusters(self, vectors_in_cluster):
        path = VECTOR_DIR_PATH
        dim = VECTOR_DIM
        size = len(os.listdir(path))
        return size//vectors_in_cluster + 1
        
    
    def initialize(self, vectors):
        kmeans = KMeans(
            init = 'random',
            n_clusters = (vectors.shape[0]//VECTORS_IN_CLUSTER + 1)
        )
        kmeans.fit(vectors)
        centroids = kmeans.cluster_centers_
        
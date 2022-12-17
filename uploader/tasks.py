import os
from config import *
import numpy as np

class KMeansImageAddTask:

    def load_centroids(self):
        centroids = {}
        for f in os.listdir(CLUSTERS_PATH):
            centroid = np.load(CLUSTERS_PATH+f).reshape(VECTOR_DIM)
            centroids[centroid[0]] = centroid[1:]
        return centroids
    
    def load_mapping(self)
        

    def execute(self, vector):
        centroids = self.load_centroids() 
        similarities = []
        for k, v in centroids:
            similarities.append((k, self.cosine_similarity(vector, v)))
        
        similarities.sort(key = lambda x: -x[1])
        
    
    def cosine_similarity(self, A, B):
        return (np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))+1)/2
        

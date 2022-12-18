import os
from config import *
import numpy as np
import pickle

class KMeansImageAddTask:

    def load_centroids(self):
        centroids = {}
        for f in os.listdir(CLUSTERS_PATH):
            centroid = np.load(CLUSTERS_PATH+f).reshape(VECTOR_DIM)
            centroids[f[0:-4]] = centroid
        return centroids
    
    def load_mappings(self):
        with open(MAPPING_PATH, 'rb') as f:
            mapping = pickle.load(f)
        rev_mapping = {}
        for k, v in mapping.items():
            if v not in rev_mapping:
                rev_mapping[v] = []
            
            rev_mapping[v].append(k)
        return (mapping, rev_mapping)
    
    def save_mapping(self, mapping):
        with open(MAPPING_PATH, 'wb') as f:
            pickle.dump(mapping, f)
    
    def save_centroid(self, name, centroid_vector):
        np.save(CLUSTERS_PATH+name, centroid_vector) # saves vector of shape (VECTOR_DIM,)

    def execute(self, vector, name):
        centroids = self.load_centroids()
        mappings, rev_mapping = self.load_mappings()
        similarities = []
        for k, v in centroids.items():
            similarities.append((k, self.cosine_similarity(vector, v)))

        if len(similarities) == 0: # the first image
            new_centroid_vector = vector
            centroid = name
            self.save_centroid(name, new_centroid_vector)
            mappings[name] = name
            self.save_mapping(mappings)
            return
        else:
            similarities.sort(key = lambda x: -x[1])
            centroid = similarities[0][0]

        if len(rev_mapping[centroid]) == MAX_CLUSTER_SIZE:
            pass
        else:
            mappings[name] = centroid
            

        
        self.save_mapping(mappings)


    def find_new_mean_of_centroid(self, array): # array of n x m will output vector of m
        return np.average(array, axis = 0)

    def cosine_similarity(self, A, B): #array cosine similarity between 2 vectors
        return (np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))+1)/2
        

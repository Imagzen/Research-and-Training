import os
from config import *
import numpy as np
import pickle
from sklearn.cluster import KMeans

class KMeansImageAddTask:

    def load_vectors(self):
        vector_array = []
        vector_to_ind = {}
        count = 0
        for file in os.listdir(VECTOR_DIR_PATH):
            vector = np.load(VECTOR_DIR_PATH+file).reshape(VECTOR_DIM)
            vector_array.append(vector)
            vector_to_ind[file[0:-4]] = count
            count+=1
        return np.stack(vector_array, axis = 0), vector_to_ind

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
    
    def find_vectors_by_name(self, vector_name, vector_to_ind, vector):
        indexes = []
        for name in vector_name:
            indexes.append(vector_to_ind[name])

        return vector[indexes]

    def execute(self, vector, name):
        centroids = self.load_centroids()
        mappings = {}
        if len(centroids)== 0: # the first image
            new_centroid_vector = vector
            centroid = name
            self.save_centroid(name, new_centroid_vector)
            mappings[name] = name
            self.save_mapping(mappings)
            print('Adding new centroid: '+centroid)
            return
        
        mappings, rev_mapping = self.load_mappings()
        similarities = []
        for k, v in centroids.items():
            similarities.append((k, self.cosine_similarity(vector, v)))

        
        similarities.sort(key = lambda x: -x[1])
        centroid = similarities[0][0]
        vectors, vector_to_ind = self.load_vectors()
        cluster_vectors = self.find_vectors_by_name(rev_mapping[centroid], vector_to_ind, vectors)
        cluster_vectors = np.concatenate((cluster_vectors, vector.reshape(1, -1)), axis = 0)

        if len(rev_mapping[centroid]) == MAX_CLUSTER_SIZE:
            new_centroid_vectors = self.split(cluster_vectors, 2)
            self.save_centroid(centroid, new_centroid_vectors[0])
            self.save_centroid(name, new_centroid_vectors[1])
            # to find which vector will belong to which centroid
            centroid_1_list = []
            centroid_2_list = []
            for vector_name in rev_mapping[centroid]:
                v = vectors[vector_to_ind[vector_name]]
                if self.cosine_similarity(new_centroid_vectors[0], v) >= self.cosine_similarity(new_centroid_vectors[1], v):
                    centroid_1_list.append(vector_name)
                else:
                    centroid_2_list.append(vector_name)
            
            for vector_name in centroid_1_list:
                mappings[vector_name] = centroid
            
            for vector_name in centroid_2_list:
                mappings[vector_name] = name
            print('Splitting centroid '+centroid)
        else:
            mappings[name] = centroid
            new_centroid_vector = self.find_new_mean_of_centroid(cluster_vectors)
            self.save_centroid(centroid, new_centroid_vector)
            print('Adding to existing centroid '+centroid)
        
        self.save_mapping(mappings)

    def split(self, vectors, n_clusters): # perform kmeans clustering
        kmeans = KMeans(n_clusters=n_clusters).fit(vectors)
        return kmeans.cluster_centers_

    def find_new_mean_of_centroid(self, array): # array of n x m will output vector of m
        return np.average(array, axis = 0)

    def cosine_similarity(self, A, B): #array cosine similarity between 2 vectors
        return (np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))+1)/2
        

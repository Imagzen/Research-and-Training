from abc import ABC, abstractmethod
import numpy as np
from config import *
import os
import pickle
from converters.text_converters import GoogleTextConverter
from similarityfunctions.similarity import CosineSimilarity
from similarityfunctions.provider import SimilarityProvider
from logger.Logger import Logger
class Searcher(ABC):

    def __init__(self, text_converter):
        self.similarity_calculator = SimilarityProvider.getSimilarityCalculator()
        self.text_converter = text_converter
        self.vectors = self.load_vectors(VECTOR_DIR_PATH, VECTOR_DIM)
        self.names = [n for n in os.listdir(IMAGE_DIR_PATH)]
    
    def load_vectors(self, path, dim):
        vector_array = []
        for file in os.listdir(path):
            vector = np.load(path+file).reshape(dim)
            vector_array.append(vector)
        return np.stack(vector_array, axis = 0)
    
    def search(self, desc):
        Logger.d("Searching", desc)
        input_vector = self.text_converter.convert(desc).reshape(VECTOR_DIM)
        scores = self.getMostSimilarVectors(input_vector, SEARCH_COUNT)
        scores.sort(key = lambda x: -1*x[1])
        result = []
        for s in scores:
            Logger.i("Results", self.names[s[0]]+", "+str(s[1]))
            result.append((self.names[s[0]], s[1]))
        
        return result

    @abstractmethod
    def getMostSimilarVectors(self, input_vector, output_size):
        pass


class LinearSearcher(Searcher): # time consuming best accuracy

    def __init__(self, text_converter):
        super().__init__(text_converter)

    def getMostSimilarVectors(self, input_vector, output_size):
        output = []
        for i in range(self.vectors.shape[0]):
            A = self.vectors[i]
            B = input_vector
            similarity = self.similarity_calculator.calculate(A, B)
            output.append((i, similarity))
        output.sort(key = lambda x: -1*x[1])
        return output[0:output_size] 

class GreedySearcher(Searcher): # time best, accuracy worst

    def __init__(self, text_converter):
        super().__init__(text_converter)

    def util(self, input_vector, output_set, column_number, thresh):
        if thresh < MIN_THRESH:
            thresh = MIN_THRESH
        
        if column_number == VECTOR_DIM:
            return output_set
        
        v = input_vector[column_number]

        dict_ = {i:[] for i in range(1,101)}
        next_output_set = []
        distances = []
        for row_number in output_set:
            
            value = self.vectors[row_number][column_number]
            x = 1
            for s in range(1, 101):
                if value>(s/100):
                    break
                else:
                    x = s
            
            dict_[x].append(row_number)
            
        for s in range(100, 0, -1):
            distances.append((abs(s/100 - v), s))
        
        
        distances.sort(key = lambda x: x[0]) 
        count = 0
        for d in distances:
            for l in dict_[d[1]]:
                next_output_set.append(l)
            count+=len(dict_[d[1]])
            if count>=thresh:
                break 
            
        return self.util(input_vector, next_output_set, column_number+1, thresh - 10)

    def getMostSimilarVectors(self, input_vector, output_size):
        output_set = [i for i in range(self.vectors.shape[0])]
        output_set = self.util(input_vector, output_set, 0, THRESH)
        output = []
        for i in output_set:
            A = self.vectors[i]
            B = input_vector
            similarity = self.similarity_calculator.calculate(A, B)
            output.append((i, similarity))
        
        output.sort(key = lambda x: -1*x[1])
        return output[0:output_size] 

class KMeansSearching(Searcher): # medium accuracy and execution time

    def __init__(self, text_converter):
        super().__init__(text_converter)
    
    def load_vectors_with_mapping(self):
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

    def getMostSimilarVectors(self, input_vector, output_size):
        centroids = self.load_centroids()
        mappings, rev_mappings = self.load_mappings()
        vectors , vector_to_ind = self.load_vectors_with_mapping()

        similarities = []
        for c in centroids.keys():
            similarity = self.similarity_calculator.calculate(centroids[c], input_vector)
            similarities.append((c, similarity))
        
        similarities.sort(key = lambda x: -x[1])
        target_centroid = similarities[0][0]
        vector_names = rev_mappings[target_centroid]
        output = []
        for name in vector_names:
            A = vectors[vector_to_ind[name]].reshape(VECTOR_DIM)
            B = input_vector.reshape(VECTOR_DIM)
            s = (np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))+1)/2
            output.append((vector_to_ind[name], s))
        
        output.sort(key=  lambda x: -x[1])
        return output
    
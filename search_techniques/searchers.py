from abc import ABC, abstractmethod
import numpy as np
from config import *
import os
from converters.text_converters import GoogleTextConverter

class Searcher(ABC):

    def __init__(self, text_converter):
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
        print("Searching "+desc)
        input_vector = self.text_converter.convert(desc).reshape(VECTOR_DIM)
        scores = self.getMostSimilarVectors(input_vector, SEARCH_COUNT)
        for s in scores:
            print(self.names[s[0]], end = ', ')
            print(s[1])

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
            similarity = (np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))+1)/2
            output.append((i, similarity))
        output.sort(key = lambda x: -1*x[1])
        return output[0:output_size] 

class GreedySearcher(Searcher): # time best, accuracy worst

    def __init__(self, text_converter):
        super().__init__(text_converter)

    def util(self, output_set, column_number):
        dict_ = {i:[] for i in range(1,101)}
        for row_number in output_set:
            if column_number == VECTOR_DIM:
                return output_set
            
            value = self.vectors[row_number][column_number]
            distances = []
            count = 0
            for s in range(100, 0, -1):
                if value<=(s/100):
                    dict_[s].append(row_number)
                    break
            
            for s in range(100, 0, -1):
                distances.append((abs(s/100 - value), s))

            distances.sort(key = lambda x: x[0]) 
            next_output_set = []
            for d in distances:
                for l in dict_[d[1]]:
                    next_output_set.append(l)
                count+=len(dict_[d[1]])
                if count>=THRESH:
                    break 
            
            return self.util(next_output_set, column_number+1)


    def getMostSimilarVectors(self, input_vector, output_size):
        output_set = set([i for i in range(self.vectors.shape[0])])
        output_set = self.util(output_set, 0)
        output = []
        for i in output_set:
            A = self.vectors[i]
            B = input_vector
            similarity = (np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))+1)/2
            output.append((i, similarity))
        
        output.sort(key = lambda x: -1*x[1])
        return output[0:output_size] 

class KMeansSearching: # medium accuracy and execution time

    def __init__(self, text_converter):
        super().__init__(text_converter)

    def getMostSimilarVectors(self, input_vector, output_size):
        pass
    

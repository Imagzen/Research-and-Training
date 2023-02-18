from abc import ABC, abstractclassmethod
import numpy as np

class Similarity(ABC):

    @abstractclassmethod
    def calculate(self, inp1, inp2):
        None

class CosineSimilarity(Similarity):

    def calculate(self, A, B):
        return (np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))+1)/2
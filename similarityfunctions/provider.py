
class SimilarityProvider:
    '''
        Used to provide the object of similarity that is being used.
        The provider is initiated with the a similarity_calculator at the start of the applications (app.py)
    '''

    similarity_calculator = None

    @classmethod
    def getSimilarityCalculator(cls):
        if cls.similarity_calculator is None:
            raise Exception('Similarity Calculator not set')

        return cls.similarity_calculator

    @classmethod
    def setSimilarityCalculator(cls, similarity_calculator):
        cls.similarity_calculator = similarity_calculator 
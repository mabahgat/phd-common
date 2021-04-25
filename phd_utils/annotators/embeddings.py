from sentence_transformers import SentenceTransformer
from scipy import spatial


class SentenceEmbeddings:

    def __init__(self, model_str='bert-base-nli-mean-tokens'):
        self.__model = SentenceTransformer(model_str)
    
    def get(self, sentence_str):
        """
        Returns sentence embeddings for input sentence
        """
        return self.__model.encode(sentence_str)
    
    @staticmethod
    def distance(emb1_arr, emb2_arr, method_str='cosine'):
        """
        Measures the distance between two embeddings
        :param method_str: measure method: cosine
        """
        if method_str == 'cosine':
            return spatial.distance.cosine(emb1_arr, emb2_arr)
        else:
            raise ValueError('Unknown method {}'.format(method_str))


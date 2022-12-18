

import numpy as np

class GoogleTextConverter:
    def __init__(self):
        import tensorflow as tf
        import tensorflow_hub as hub
        from config import GOOGLE_EMBEDDINGS_TF_HUB_URL
        with tf.device('/CPU:0'):
            self.embed = hub.load(GOOGLE_EMBEDDINGS_TF_HUB_URL) # works on RNN LSTMS

    def convert(self, desc):
        arr = np.array(self.embed([desc]))
        return arr


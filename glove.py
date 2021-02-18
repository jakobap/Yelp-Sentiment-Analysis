import numpy as np

path_to_glove_file = "./embeddings/glove.6B.300d.txt"

embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("%s Vectors Imported from Glove." % len(embeddings_index))

import numpy as np

path_to_glove_file = "embeddings/glove.6B.300d.txt"

pretrained_embedd = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        pretrained_embedd[word] = coefs

print("%s Vectors Imported from Glove." % len(pretrained_embedd))

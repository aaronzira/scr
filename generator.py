import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cosine,chebyshev,canberra,euclidean
from pyemd import emd

class DataGenerator():
    """
    Data Generator class used to generate numpy binaries ("*.npy") of data for model testing on
    phrases differences. Specify location of word embeddings (should be size 3M x 300), location
    of training set, and length of desired output vectors:
    600 for just sum vectors, 602 for sums and wmd distances, 604 for sums and four distance metrics,
    or 606 for all of the above.
    """

    def __init__(self,embeddings_path,training_set,vector_len):

        self.data_path = embeddings_path
        self.training_set = training_set
        self.vec_len = vector_len
        self.epsilon = 1e-4

        binary_file = os.path.join(self.data_path,'GoogleNews-vectors-negative300.bin')
        w2v_dat = os.path.join(self.data_path,'embed.dat')
        w2v_vocab = os.path.join(self.data_path,'embed.vocab')

        # create word embeddings and mapping of vocabulary item to index
        self.embeddings = np.memmap(w2v_dat, dtype=np.float64,mode="r", shape=(3000000, 300))

        with open(w2v_vocab) as f:
            vocab_list = map(lambda string: string.strip(), f.readlines())
        self.vocab_dict = {w: i for i, w in enumerate(vocab_list)}

    def _get_dist(self,s_1,s_2,distances,wmd):

        results_ = []

        ### will need for predictions, but not for generation:
        #s_1 = re.sub(r'-',' ',s_1)
        #s_1 = re.sub(r"\ba\b",'one',s_1)
        #s_1 = re.sub(r'_+','',s_1)
        #s_1 = re.sub(r"'cause",'because',s_1)

        #s_2 = re.sub(r'-',' ',s_2)
        #s_2 = re.sub(r"\ba\b",'one',s_2)
        #s_2 = re.sub(r'_+','',s_2)
        #s_2 = re.sub(r"'cause",'because',s_2)

        # also what to do if either string is now blank

        # actual words
        s1_features = s_1.split()
        s2_features = s_2.split()

        # sum (OR mean) of word embeddings per string
        # no longer need for OOV or blank checks
        # shape is [1,300] per string
        S1_ = self.embeddings[[self.vocab_dict[w] for w in s1_features]]
        S2_ = self.embeddings[[self.vocab_dict[w] for w in s2_features]]

        # CHANGED THIS TO MEAN
        S1_ = np.asarray(np.mean(S1_,axis=0)).reshape([-1,1])+1e-50
        S2_ = np.asarray(np.mean(S2_,axis=0)).reshape([-1,1])+1e-50
        #diff = S2_-S1_
        results_.extend(S1_)
        results_.extend(S2_)
        #results.extend(diff)

        if distances:
            results_.append(canberra(S1_,S2_))
            results_.append(chebyshev(S1_,S2_))
            results_.append(cosine(S1_,S2_))
            results_.append(euclidean(S1_,S2_))

        if wmd:
            # fit CV on words with or without a single quote
            vect = CountVectorizer(token_pattern='[\w\']+').fit([s_1, s_2])
            features = np.asarray(vect.get_feature_names())
            W_ = self.embeddings[[self.vocab_dict[w] for w in features]]

            # get 'flow' vectors
            v_1, v_2 = vect.transform([s_1, s_2])
            v_1 = v_1.toarray().ravel().astype(np.float64)
            v_2 = v_2.toarray().ravel().astype(np.float64)

            # normalize vectors so as not to reward shorter strings in WMD
            v_1 /= (v_1.sum()+self.epsilon)
            v_2 /= (v_2.sum()+self.epsilon)

            # use both euclidean and cosine dists (cosine dist is 1-cosine sim)
            D_euclidean = euclidean_distances(W_).astype(np.float64)
            D_cosine = 1.-cosine_similarity(W_,).astype(np.float64)

            # using EMD (Earth Mover's Distance) from PyEMD
            distances_euclidean = emd(v_1,v_2,D_euclidean)
            distances_cosine = emd(v_1,v_2,D_cosine)

            # both WMD calculations (euclidean and cosine)
            results_.append(distances_euclidean)
            results_.append(distances_cosine)

        return np.asarray(results_,dtype=np.float).reshape([-1,self.vec_len])

    def generate(self,output_directory,distances,wmd,pos_tags,seed=30):
        """
        Generate numpy binaries. Specify output directory, True/False for whether or not to
        use distance metrics and/or wmd calculations, and optionally random seed.
        """

        # original training set cols are Error_type, Str_1, Str_2
        X_in = np.genfromtxt(self.training_set,
                      delimiter=',',usecols=(1,2),dtype=str)
        Y_in = np.genfromtxt(self.training_set,
                      delimiter=',',usecols=(0)).reshape((-1,1))

        X = []
        Y = []

        if pos_tags:
            X_tags = np.genfromtxt(self.training_set,
                                   delimiter=',',usecols=(3,4))


            for i,strings in enumerate(X_in):
                scores = self._get_dist(strings[0],strings[1],distances,wmd)
                scores = np.append(scores,X_tags[i])
                X.extend(scores)

                # target
                Y.append(Y_in[i])
            X = np.asarray(X).reshape((-1,self.vec_len+2))

        else:
            for i,strings in enumerate(X_in):
                scores = self._get_dist(strings[0],strings[1],distances,wmd)
                X.extend(scores)

                # target
                Y.append(Y_in[i])


            X = np.asarray(X).reshape((-1,self.vec_len))

        Y = np.asarray(Y).reshape((-1,1))

        # unshuffled indices
        indices = range(X.shape[0])

        # randomly shuffle the data
        np.random.seed(seed)
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]

        # transform Y from either 1 or 2 to a one-hot vector ([1,0] or [0,1])
        y_list = []
        for i, label in enumerate(Y):
            if label == 2:
                label = 1
                y_list.append(np.insert(label,0,0))
            elif label == 1:
                y_list.append(np.insert(label,1,0))
            else:
                raise ValueError("Y label must be either 1 (minor) or 2 (major). Problem at index ", indices[i])
        Y = np.asarray(y_list)

        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
        self.X_in = X_in
        self.Y_in = Y_in.astype(np.float32)
        self.indices = np.asarray(indices)

        arrays = {'X':self.X,'Y':self.Y,'indices':self.indices,'X_in':self.X_in,'Y_in':self.Y_in}
        for name,item in arrays.items():
            np.save('{}/{}'.format(output_directory,name),item)









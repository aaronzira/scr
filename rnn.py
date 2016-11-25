import tensorflow as tf
import numpy as np
import os
import re
from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cosine, euclidean, chebyshev, canberra
from pyemd import emd
import matplotlib.pyplot as plt

import nltk
from nltk.tag import map_tag,pos_tag


class Model():

    def __init__(self,input_len,distances,pos_tags,wmd,lr=0.001,n_iters=1000,mb_size=32,disp_step=1,n_timesteps=1,num_hidden=128,n_class=2):
        self.data_path = "./"
        self.distances = distances
        self.wmd = wmd
        self.pos_tags = pos_tags
        self.epsilon = 1e-4

        binary_file = os.path.join(self.data_path,
                                   'GoogleNews-vectors-negative300.bin')
        w2v_dat = os.path.join(self.data_path,'embed.dat')
        w2v_vocab = os.path.join(self.data_path,'embed.vocab')

        # create word embeddings and mapping of vocabulary item to index
        self.embeddings = np.memmap(w2v_dat, dtype=np.float64,
                                    mode="r", shape=(3000000, 300))
        with open(w2v_vocab) as f:
            vocab_list = map(lambda string: string.strip(), f.readlines())

        self.vocab_dict = {w: i for i, w in enumerate(vocab_list)}

        # Net parameters
        self.learning_rate = lr
        self.training_iters = n_iters
        self.batch_size = mb_size
        self.display_step = disp_step
        self.n_input = input_len#13#608
        self.n_steps = n_timesteps # timesteps
        self.n_hidden = num_hidden#128 # hidden layer num of features
        self.n_classes = n_class


        df_tags = np.asarray([[u'ADJ'], [u'ADJ', u'VERB'], [u'ADP'], [u'ADV'], [u'ADV', u'PRT'],
       [u'ADV', u'VERB'], [u'CONJ'], [u'DET'], [u'DET', u'VERB'],
       [u'NOUN'], [u'NOUN', u'.'], [u'NOUN', u'NOUN'], [u'NOUN', u'PRT'],
       [u'NOUN', u'VERB'], [u'NUM'], [u'NUM', u'PRT'], [u'PRON'],
       [u'PRON', u'.'], [u'PRON', u'VERB'], [u'VERB'], [u'VERB', u'ADV'],
       [u'VERB', u'NOUN'], [u'VERB', u'VERB']], dtype=object)
        self.tag_dict = {k:v for v,k in enumerate([str(tag) for tag in df_tags])}


        # mean of 20 rarest words, used as a stand-in for pairwise distances
        # if a word is out-of-vocabulary
        ##avg_rare_word = np.mean(np.vstack(embeddings[-20:]),axis=0)
        ##bad_row = np.asarray([avg_rare_word])


    def data_gen(self,val_size=.2,tes_size=.1):

        # reshapes really just serve to warn me when i'm being dumb
        ##X = np.fromfile('./X',dtype=np.float32)
        X = np.load('./X.npy')
        self.X = X.reshape([-1,self.n_input])
        ##Y = np.fromfile('./Y',dtype=np.float32)
        Y = np.load('./Y.npy')
        self.Y = Y.reshape([-1,self.n_classes])

        ##self.indices = np.fromfile('./indices',dtype=np.int)
        self.indices = np.load('./indices.npy')
        ##X_in = np.fromfile('./X_in',dtype=x_in_type)
        X_in = np.load('./X_in.npy')
        self.X_in = X_in.reshape([-1,2])
        ##Y_in = np.fromfile('./Y_in',dtype=np.float32)
        Y_in = np.load('./Y_in.npy')
        self.Y_in = Y_in.reshape([-1,1])

        validation_size = val_size
        test_size = tes_size

        # create split indices for validation, test, and train sets
        self._validation_test_split_idx = int(len(self.Y)*validation_size)
        self._train_test_split_idx = int(len(self.Y)*test_size)+self._validation_test_split_idx

        # split data
        self.x_validation = self.X[:self._validation_test_split_idx]
        self.x_test = self.X[self._validation_test_split_idx:
                             self._train_test_split_idx]
        self.x_train = self.X[self._train_test_split_idx:]
        self.y_validation = self.Y[:self._validation_test_split_idx]
        self.y_test = self.Y[self._validation_test_split_idx:
                             self._train_test_split_idx]
        self.y_train = self.Y[self._train_test_split_idx:]
        #print(x_validation.shape,x_test.shape,x_train.shape)


    def RNN(self, x, weights, biases):

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # Reshaping to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, self.n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0, self.n_steps, x)

        # Define a lstm cell with tensorflow
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden, state_is_tuple=True)

        # Get lstm cell output
        outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32)
        ##outputs = tf.Print(outputs,[outputs],"OUTPUTS: ")

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']


    def build_graph(self):
        '''
        '''

        # tf Graph input
        x = tf.placeholder("float", [None, self.n_steps, self.n_input])
        y = tf.placeholder("float", [None, self.n_classes])

        # Define weights
        weights = {
            'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        pred = self.RNN(x, weights, biases)

        # Define loss and optimizer
        #########J = sum(sum(6*(y*tf.log(pred))+(1-y)*tf.log(1-pred)))

        cost_all = tf.nn.softmax_cross_entropy_with_logits(pred, y)
        #cost_all = tf.Print(cost_all,[cost_all],"cost_all: ")

        cost = tf.reduce_mean(cost_all)
        ##ratio = 8034.0 / (66561.0 + 8034.0)
        #class_weight = tf.constant(1.-ratio)#, 1.0 - ratio
        ##class_weight = tf.constant(-ratio)
        ##cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(pred, y, pos_weight=class_weight))
        ### doesn't work but cost = tf.reduce_mean(tf.square(tf.cast(tf.less(pred, y),dtype=tf.float32)))

        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        return (x,y),cost,optimizer,pred,accuracy,self.learning_rate,tf.train.Saver()


    def train(self):
        tf.reset_default_graph()
        (X,Y),cost,train_op,preds,accuracy,lr,saver = self.build_graph()

        x_train_reshaped = self.x_train.reshape((-1,self.n_steps,self.n_input))
        x_val_reshaped = self.x_validation.reshape((-1,self.n_steps,self.n_input))
        x_test_reshaped = self.x_test.reshape((-1,self.n_steps,self.n_input))
        self.val_acc_list = [0]
        self.tr_acc_list = [0]
        self.val_cost_list = [0]
        self.tr_cost_list = [0]

        # Launch the graph
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            #print(preds.eval({X:x_train_reshaped})) returns numbers
            #print(cost.eval({X:x_train_reshaped,Y:self.y_train})) # NAN

            start_end = zip(range(0,len(self.x_train),self.batch_size),
                       range(self.batch_size,len(self.x_train)+1,
                             self.batch_size))

            for pass_i in range(self.training_iters):
                for (s,e) in start_end:

                    ######x_batch = x_train[s:e,].reshape((batch_size,n_steps,n_input))
                    # Run optimization op (backprop)
                    sess.run(train_op, feed_dict={X: x_train_reshaped[s:e,:,:], Y: self.y_train[s:e]})



                    #print(s,any(np.isnan(preds.eval({X:x_train_reshaped[s:e,:,:]}).ravel())))
                    #STARTING AT 12416 there are NANS
                    #if any(np.isnan(preds.eval({X:x_train_reshaped[s:e,:,:]}).ravel())):
                    #    print(s,": \nX:",x_train_reshaped[s:e,:,:])
                        #print(np.log(x_train_reshaped[s:e,:,:]))
                    #   number = input()
                    #if s > 12360:
                    #    print(np.min(x_train_reshaped[s:e,:,:],axis=2))
                    #    #print(np.max(x_train_reshaped[s:e,:,:],axis=2))
                    #    print(x_train_reshaped[s+4,:,:])
                    #    number = input()




                    # Calculate batch accuracy
                    #train_acc = sess.run(accuracy, feed_dict={x: x_batch, y: y_train[s:e]})
                    # Calculate batch loss
                    #train_loss = sess.run(cost, feed_dict={x: x_batch, y: y_train[s:e]})

                #print('validation accuracy: ',sess.run(accuracy, feed_dict={X: x_val_reshaped, Y: y_validation}))

                tr_loss = sess.run(cost, feed_dict={X: x_train_reshaped, Y: self.y_train})
                tr_acc = sess.run(accuracy, feed_dict={X: x_train_reshaped, Y: self.y_train})
                val_loss = sess.run(cost, feed_dict={X: x_val_reshaped, Y: self.y_validation})
                val_acc = sess.run(accuracy, feed_dict={X: x_val_reshaped, Y: self.y_validation})

                if pass_i % 10 == 0:
                    print("Iter " + str(pass_i) + ", Train Loss= " + \
                          "{:.6f}".format(tr_loss) + ", Train Acc= " + \
                          "{:.5f}".format(tr_acc) + ", Val Loss= " + \
                          "{:.6f}".format(val_loss) + ", Val Acc= " + \
                          "{:.5f}".format(val_acc)
                         )
                if val_acc > max(self.val_acc_list):
                    saver.save(sess,'./model.ckpt')
                    print('model saved.')

                self.val_acc_list.append(val_acc)
                self.tr_acc_list.append(tr_acc)
                self.val_cost_list.append(val_loss)
                self.tr_cost_list.append(tr_loss)


        lists = {'val_acc':self.val_acc_list,'train_acc':self.tr_acc_list,'val_cost':self.val_cost_list,'train_cost':self.tr_cost_list}
        for name,goods in lists.items():
            with open('./{}.txt'.format(name), 'w') as f:
                for item in goods:
                    f.write("%s\n" % item)

        print("Optimization Finished!")



    def _simple_tagger(self,string):
        tokens = nltk.word_tokenize(string)
        tagged = pos_tag(tokens)
        simplified = [(map_tag('en-ptb', 'universal', tag)) for word, tag in tagged]
        key = str(simplified)

        return self.tag_dict[key]


    def _get_dist(self,s_1,s_2):


        s1_features = s_1.split()
        s2_features = s_2.split()


        S1_ = self.embeddings[[self.vocab_dict[w] for w in s1_features]]
        S2_ = self.embeddings[[self.vocab_dict[w] for w in s2_features]]

        S1_sum_ = np.asarray(np.sum(S1_,axis=0)).reshape([-1,1])+1e-50
        S2_sum_= np.asarray(np.sum(S2_,axis=0)).reshape([-1,1])+1e-50

        results_ = np.concatenate((S1_sum_,S2_sum_))

        if self.distances:
            can = canberra(S1_sum_,S2_sum_)
            cheb = chebyshev(S1_sum_,S2_sum_)
            cos = cosine(S1_sum_,S2_sum_)
            euc = euclidean(S1_sum_,S2_sum_)

            results_ = np.append(results_,[can,cheb,cos,euc])



        if self.wmd:
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
            results_ = np.append(results_,[distances_euclidean,distances_cosine])

        if self.pos_tags:
            s1_tag = self._simple_tagger(s_1)
            s2_tag = self._simple_tagger(s_2)
            results_ = np.append(results_,[s1_tag,s2_tag])


        #return np.asarray(results_,dtype=np.float).reshape([-1,self.n_input])
        results_ = results_.reshape([-1,self.n_input])

        return results_


    def predict(self,numpy_file):
        predictions = []
        # build graph and initialize session
        tf.reset_default_graph()

        (X,_),_,_,pred_y,_,_,saver = self.build_graph()

        #### testing influence on predictions
        pred_y = pred_y + tf.constant([1.,0])

        with tf.Session() as sess:

            sess.run(tf.initialize_all_variables())
            saver.restore(sess,'./model.ckpt')
            # generate calculations from 2d array of input strings
            for row in numpy_file:
                str_1,str_2 = row[0],row[1]

                # strings identical
                if str_1 == str_2:
                    predictions.append('No error')
                    continue

                # model prediction
                try:
                    #tf.arg_max(pred_y,1),
                    pred = sess.run([tf.argmax(pred_y,1)],
                                    feed_dict=\
                                    {X: np.asarray(self._get_dist(str_1,str_2)).reshape([-1,self.n_steps,self.n_input])})
                    #print(pred)
                    #predictions.append(str(pred[0][0]+1))
                    predictions.append(pred[0][0]+1)

                # can't predict
                except:
                    predictions.append('Unknown')

        return predictions

    def check(self,percent=1.):

        max_preds = int(len(self.y_test)*percent)
        raw_x_test = self.X_in[self.indices[self._validation_test_split_idx:self._train_test_split_idx]]
        preds = self.predict(raw_x_test[:max_preds])
        y_test_res = np.argmax(self.y_test[:max_preds],1)+1
        no_errors = np.where(np.asarray(preds)=='No error')[0].tolist()
        unknowns = np.where(np.asarray(preds)=='Unknown')[0].tolist()
        for i in no_errors:
            preds[i] = 1
        for i in unknowns:
            preds[i] = -1

        #print(sum([val==-1 for val in preds]))

        return confusion_matrix(y_test_res,preds)

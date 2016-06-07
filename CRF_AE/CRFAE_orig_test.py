import numpy as np
import itertools
import rtk
from collections import defaultdict as dd

class CRFAE:
    '''
    Simple implementation for testing CRF AE original for small sequences
    - Assumes a sequence of size 4
    '''

    def feature_functions(self):
        return {
                1: [
                    lambda i, x, z1: ('single', x[i], z1),
                    #lambda i, x, z1: ('single', self.fgen.hypthen(x[i]), z1),
                    #lambda i, x, z1: ('single', self.fgen.digit(x[i]), z1),
                    #lambda i, x, z1: ('single', self.fgen.prefix2(x[i]), z1),
                    #lambda i, x, z1: ('single', self.fgen.prefix3(x[i]), z1),
                    #lambda i, x, z1: ('single', self.fgen.suffix2(x[i]), z1),
                    #lambda i, x, z1: ('single', self.fgen.suffix3(x[i]), z1),
                    #lambda i, x, z1: ('single', self.fgen.freq(x[i]), z1),
                    #lambda i, x, z1: ('single', self.fgen.shape(x[i]), z1),
                    ],
                3: [
                    lambda i, x, z1, z2, z3: ('triple', z1, z2, z3),
                    ],
                }


    def __init__(self, feature_generator, latent_states=range(12), window_size=3):
        self.weights = dd(float)
        self.fgen = feature_generator
        self.latent_states = latent_states
        self.window_size = window_size

    def update_sentence(self, sentence):
        self.sentence = sentence
        self.calc_partition()

    def calc_partition(self):
        lsize = len(self.latent_states)
        pzx = np.zeros((lsize, lsize, lsize, lsize))

        # Unary weights
        for latent_states in itertools.product(self.latent_states, repeat=4):
            for i in range(len(latent_states)):
                for fcn in self.feature_functions()[1]:
                    pzx[latent_states] += self.weights[fcn(i, self.sentence, latent_states[i])]

        # Triple weights
        for latent_states in itertools.product(self.latent_states, repeat=4):
            for fcn in self.feature_functions()[3]:
                # First window
                pzx[latent_states] += self.weights[fcn(0, self.sentence, *latent_states[0:3])]
                # Second window
                pzx[latent_states] += self.weights[fcn(1, self.sentence, *latent_states[1:4])]

        self.partition = pzx

    def logZ(self):
        return rtk.m.logsumexp(self.partition)

    def logZ_config(self, (i, latent_states)):
        if i == 0:
            partition = np.sum(np.exp(self.partition), axis=3)
        elif i == 1:
            partition = np.sum(np.exp(self.partition), axis=0)
        return np.log(partition[latent_states])

    def compute_objective(self, generative_model):
        unnormalized_obj = np.copy(self.partition)
        for latent_states in itertools.product(self.latent_states, repeat=4):
            for i in range(4):
                unnormalized_obj[latent_states] += np.log(generative_model.px_given_z(self.sentence[i], latent_states[i]))
        return rtk.m.logsumexp(unnormalized_obj) - self.logZ()


    def compute_gradient(self, generative_model):
        gradient = dd(float)
        recprob = np.exp(self.partition) / np.sum(np.exp(self.partition))
        jointprob = np.copy(recprob)
        for latent_states in itertools.product(self.latent_states, repeat=4):
            for i in range(4):
                jointprob[latent_states] *= generative_model.px_given_z(self.sentence[i], latent_states[i])
        jointprob /= np.sum(jointprob)

        # Calculate feature expectation over q(z|x)
        feat_exp = dd(float)
        for latent_states in itertools.product(self.latent_states, repeat=4):
            for fcn in self.feature_functions()[1]:
                for i in range(4):
                    feat_exp[fcn(i, self.sentence, latent_states[i])] += recprob[latent_states]
            for fcn in self.feature_functions()[3]:
                feat_exp[fcn(0, self.sentence, *latent_states[0:3])] += recprob[latent_states]
                feat_exp[fcn(1, self.sentence, *latent_states[1:4])] += recprob[latent_states]

        # Calculate the gradient
        for latent_states in itertools.product(self.latent_states, repeat=4):
            #print latent_states, jointprob[latent_states]
            temp = dd(float)
            for fcn in self.feature_functions()[1]:
                for i in range(4):
                    temp[fcn(i, self.sentence, latent_states[i])] += 1
            for fcn in self.feature_functions()[3]:
                temp[fcn(0, self.sentence, *latent_states[0:3])] += 1
                temp[fcn(1, self.sentence, *latent_states[1:4])] += 1
            for feature in feat_exp:
                gradient[feature] += jointprob[latent_states] * (temp[feature] - feat_exp[feature])

        return gradient

    def numeric_gradient(self, generative_model):
        gradient = {}
        h = 1e-8
        for feature in self.weights:
            self.weights[feature] -= h
            self.calc_partition()
            low = self.compute_objective(generative_model)
            self.weights[feature] += 2* h
            self.calc_partition()
            high = self.compute_objective(generative_model)
            self.weights[feature] -= h
            self.calc_partition()
            gradient[feature] = (high - low) / (2 * h)
        return gradient
            
    # Returns MAP assignment and associated log prob
    def predict(self, generative_model):
        unnormalized_obj = np.copy(self.partition)
        for latent_states in itertools.product(self.latent_states, repeat=4):
            for i in range(4):
                unnormalized_obj[latent_states] += np.log(generative_model.px_given_z(self.sentence[i], latent_states[i]))
        inds = np.where(unnormalized_obj == np.max(unnormalized_obj))
        if len(inds[0]) > 1:
            print 'Warning - MAP assignment is not unique...'
        return tuple([inds[i][0] for i in range(4)]), np.max(unnormalized_obj) - self.logZ()


    def gradient_step(self, stepsize, g):
        for k, v in g.iteritems():
            self.weights[k] += stepsize * v
        self.calc_partition()



#---------------------- unused functions ---------------


    def _var_bound(self, seqX, seqB):
        '''
        Calculates variation lower bound -KL(q(z|x) || p(z)) + E_q(z|x)[log p(x|z)]
        INPUTS : seqX - word_features (sequence_size, feature_size)
                 seqB - brown clusters for each word (sequence_size)
        OUTPUTS : float array denoting reconstruction probability value P(hat(x),y1,y2,y3 | x_i)
        '''
        assert seqX.shape == (self.opts.SEQ_SIZE, self.opts.FEAT_SIZE)
        assert seqB.shape[0] == self.opts.SEQ_SIZE

        lsize = self.opts.LATENT_SIZE
        qzx = np.zeros((lsize, lsize, lsize)) # q(z|x)
        for (i,j,k) in itertools.product(range(lsize), repeat=3):
            sentence_potential = 0
            for seq in range(1,4):
                # First tag is a unique start tag
                hidden = (lsize,i,j,k)
                features = self._get_features(hidden[seq-1], hidden[seq], seqX, seq-1)
                sentence_potential = np.dot(features, self.encoder_weights)
            qzx[i,j,k] = np.exp(sentence_potential)
        norm = np.sum(qzx)
        qzx = qzx / norm
        
        pxz = np.zeros((lsize, lsize, lsize)) # p(x|z)
        pz = np.zeros((lsize, lsize, lsize)) # p(z)
        for (i,j,k) in itertools.product(range(lsize), repeat=3):
            pz[i,j,k] = np.prod([self.posdistr[pos_ind] for pos_ind in [i,j,k]])
            # brown clusters
            generative_prob = 1
            generative_prob *= self.decoder_weights[i,seqB[0]]
            generative_prob *= self.decoder_weights[j,seqB[1]]
            generative_prob *= self.decoder_weights[k,seqB[2]]
            pxz[i,j,k] = generative_prob

        # KL(q(z|x) || p(z))
        kl_div = qzx * np.log(qzx / pz)
        # E_q(z|x)[p(x|z)]
        ex_gen = qzx * np.log(pxz)
        
        #print np.sum(kl_div), np.sum(ex_gen), np.sum(qzx), np.min(pxz)
        return ex_gen - kl_div

    def _get_loss(self, wfeatures, wclusters):
        '''
        Calculates the model loss based on a batch of training examples
        INPUTS : wfeatures - training feature sequences (batch_size, sequence_size, feature_size)
                 wclusters - associated brown clusters (batch_size, sequence_size)
        OUTPUT : float denoting loss objective
        '''
        px = []
        for i in range(self.opts.BATCH_SIZE):
            rprob = self._var_bound(wfeatures[i,::], wclusters[i,::])
            px.append(np.sum(rprob))
        loss = np.sum(px) / self.opts.BATCH_SIZE
        # Potential regularization losses
        #loss += np.sum(np.square(self.encoder_weights))
        #loss += np.sum(np.square(self.decoder_weights))
        return loss

    # Implemented : (y_i,y_i-1) (y_i, sub_j(x_i))
    # Remaining : (y_i, sub_j(x_i), sub_j(x_i-1)) (y_i, sub_j(x_i), sub_j(x_i+1))
    def _get_features(self, yp, yc, seqX, i):
        '''
        Generates feature vector given input values g(y_i, y_i-1, x, i)
        INPUTS : yp - int denoting index of latent state of prev word in sequence
                 yc - int denoting index of latent state of current word in sequence
                 X - feature vector for all words in sequence (seq_size, feat_size)
                 i - index of current word in sequence
        OUTPUTS : feature vector
        '''
        all_pairs = itertools.product(range(self.opts.LATENT_SIZE),range(self.opts.LATENT_SIZE))
        pairF = []
        for (p1,p2) in all_pairs:
            pairF.append(int(yp == p1 and yc == p2))
        pairF = np.array(pairF)

        interF = [(yc == pos) * seqX[i,::] for pos in range(self.opts.LATENT_SIZE)]
        interF = np.array(interF).flatten()

        return np.append(pairF, interF)

if __name__ == '__main__':
    import sys
    sys.path.append("..")
    from main import GlobalOpts    

    opts = GlobalOpts()
    bcounts = np.random.rand(opts.LATENT_SIZE, opts.BCLUSTER_SIZE)
    model = CRFAE(opts, bcounts)

    wfeatures = np.random.rand(opts.SEQ_SIZE, opts.FEAT_SIZE)
    wclusters = np.random.randint(opts.BCLUSTER_SIZE, size=opts.SEQ_SIZE)
    bound = model._var_bound(wfeatures, wclusters)
    #print bound
    print 'Bound : %f' % np.sum(bound)

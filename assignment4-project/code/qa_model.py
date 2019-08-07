from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
#from tensorflow.python.ops import variable_scope as vs

from evaluate import exact_match_score, f1_score
#
import util
FLAGS = tf.app.flags.FLAGS
from os.path import join as pjoin
#
logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

    def encode(self, inputs, masks, lens, sen, sen_mask_eq,
               encoder_state_input):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """

        d_emb, q_emb = inputs
        (d_maskih, q_maskih,
         d_maski_,
         _d_maski) = masks
        # i: 0./-np.inf mask
        # ih: maski, row of zero prepended, repeat dim at 1
        d_lens_ph, q_lens_ph, d_lens_1 = lens
        d_sen_m, q_sen_m = sen
        d_sen_mask_eq, q_sen_mask_eq = sen_mask_eq

        # document rnn
        #cellf = tf.nn.rnn_cell.GRUCell(FLAGS.state_size,
        #        kernel_initializer=tf.orthogonal_initializer(),
        #        bias_initializer=tf.ones_initializer())
        cell = tf.nn.rnn_cell.LSTMCell(FLAGS.state_size,
                                       initializer=tf.orthogonal_initializer())
        d_outs, _ = tf.nn.dynamic_rnn(cell,
                                      d_emb,
                                      d_lens_ph,
                                      dtype=tf.float32,
                                      scope='rnn_doc')
                                      # sub scope is created even w/o this
        dq_tail = tf.zeros((tf.shape(d_emb)[0], 1, FLAGS.state_size),
                           name='dq_tail')
        d_outs = tf.concat((d_outs, dq_tail), 1, 'd_out_1')

        # 34362193. needed fancy tile when used to append sen to d_outs

        d_outs = tf.where(d_sen_mask_eq, d_sen_m, d_outs, 'd_out_where')
        DD = tf.transpose(d_outs, perm=[0, 2, 1], name='DD')

        # question rnn
        q_outs, _ = tf.nn.dynamic_rnn(cell,  # same cell as before
                                      q_emb,
                                      q_lens_ph,
                                      dtype=tf.float32,
                                      scope='rnn_qst')
        q_outs = tf.concat((q_outs, dq_tail), 1, 'q_out_1')
        q_outs = tf.where(q_sen_mask_eq, q_sen_m, q_outs, 'q_out_where')

        # 3d 2d matmul: 38235555
        WQ = tf.get_variable('WQ',
                             (FLAGS.state_size,)*2,
                             tf.float32,
                             tf.glorot_uniform_initializer())
        # regularizer=tf.contrib.layers.l2_regularizer(FLAGS.q_reg)
        # investigating nan in cellf
        bQ = tf.get_variable('bQ',
                             (FLAGS.q_max_width+1, FLAGS.state_size),
                             tf.float32,
                             tf.zeros_initializer())

        QQr = tf.reshape(q_outs, (-1, FLAGS.state_size), 'q_outs_reshape')
        QW = tf.matmul(QQr, WQ, name='QW')
        QWr = tf.reshape(QW, (-1, FLAGS.q_max_width+1, FLAGS.state_size),
                         'QWr')
        QT = tf.tanh(tf.add(QWr, bQ, name='QWrbQ'), name='QT')
        QQ = tf.transpose(QT, [0, 2, 1], 'QQ')

        # coattention        
        LL = tf.matmul(d_outs, QQ, name='LL')
        LT = tf.transpose(LL, perm=[0, 2, 1], name='LT')

        # mask LL, LT to hack softmax to accommodate variable lengths
        
        #adding inf to force inf is a bad idea. terrible grad might still flow
        #LL = tf.add(LL, q_maskih, 'LL_m')

        #slice assign only works with variables. 39157723
        #https://www.tensorflow.org/api_docs/python/tf/assign

        isinf_q_maskih = tf.is_inf(q_maskih, 'isinfLL')
        isinf_d_maskih = tf.is_inf(d_maskih, 'isinfLT')

        LL = tf.where(isinf_q_maskih, q_maskih, LL, 'whereLL')
        LT = tf.where(isinf_d_maskih, d_maskih, LT, 'whereLT')
        
        AQ = tf.nn.softmax(LL, dim=2, name='AQ') # has nan. necessary
        AD = tf.nn.softmax(LT, dim=2, name='AD')

        AQ0s = tf.zeros(tf.shape(AQ, 'AQshape'), tf.float32, 'AQ0s')
        AD0s = tf.zeros(tf.shape(AD, 'ADshape'), tf.float32, 'AD0s')

        AQ = tf.where(isinf_q_maskih, AQ0s, AQ, 'whereAQ') # to remove nan
        AD = tf.where(isinf_d_maskih, AD0s, AD, 'whereAD')

        CQ = tf.matmul(DD, AQ, name='CQ')
        QCQ = tf.concat((QQ, CQ), 1, name='QCQ')
        CD = tf.matmul(QCQ, AD, name='CD')
        DCD = tf.concat((DD, CD), 1, name='DCD')
        DCDT = tf.transpose(DCD, perm=[0, 2, 1], name='DCDT')

        celf = tf.nn.rnn_cell.LSTMCell(FLAGS.state_size,
                                       initializer=tf.orthogonal_initializer())
        celb = tf.nn.rnn_cell.LSTMCell(FLAGS.state_size,
                                       initializer=tf.orthogonal_initializer())
        if FLAGS.suppress_sen_in_U:
            d_lens_to_use = d_lens_ph
        else:
            d_lens_to_use = d_lens_1

        coa_outs, _ = tf.nn.bidirectional_dynamic_rnn(celf,
                                                      celb,
                                                      DCDT,
                                                      d_lens_to_use,
                                                      dtype=tf.float32,
                                                      scope='rnn_coa')
        coa_outs = tf.concat(coa_outs, 2, 'coa_outs')
        #coa_outs = coa_outs[:, :-1, :]
        UU = tf.transpose(coa_outs, perm=[0, 2, 1], name='UU')
        return UU, coa_outs, d_maski_, _d_maski, d_lens_ph


class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, knowledge_rep):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """
        UU, UT, d_maski_, _d_maski, d_lens_ph = knowledge_rep

#        # want us[i] = UU[i, :, si[i]]
#        # these don'work:
#        UUu = tf.unstack(UU, name='UU_unstacked')
#
#        us = tf.gather(cot_outs, si, axis=2, name='us')
#
#        usue = [tf.concat((UU[i, :, si[i]], UU[i, :, ei[i]]), 0)\
#                for i in xrange(tf.shape(UU)[0])]
#        # first successful try. slow
#        def us_fcn(si, UU, text_i):  # 44690865
#            us = tf.TensorArray(tf.float32, 0, True, name=text_i+'TA')
#            def cond(sample_idx, us, UU, si):
#                return tf.less(sample_idx, tf.shape(UU)[0])
#            def body(sample_idx, us, UU, si):
#                us = us.write(sample_idx, UU[sample_idx, :, si[sample_idx]])
#                return sample_idx+1, us, UU, si
#            _, us, _, _, = tf.while_loop(cond,
#                                         body,
#                                         (tf.constant(0), us, UU, si),
#                                         name='while'+text_i)
#            us = us.stack(name=text_i)
#            #print(tf.get_variable_scope().name, 3)
#            return us
#        def alph_fcn(UU, hh, us, ue, text, text_i, reuse):
#            # first version. should be correct
#            # slows down (1) building of decoder and loss (2) training
#            alph = []
#            for t in xrange(FLAGS.d_max_width):
#                inputs = tf.concat((UU[:, :, t], hh, us ,ue),
#                                   1, text_i+'_'+str(t))
#                temp1 = tf.layers.dense(inputs,
#                                        FLAGS.state_size,
#                                        activation=tf.nn.relu,
#                                        #kernel_initializer=\
#                                        #    tf.glorot_normal_initializer(),
#                                        name=text+'_1st_layer',
#                                        reuse=reuse)
#                temp2 = tf.layers.dense(temp1,
#                                        1,
#                                        #kernel_initializer=\
#                                        #    tf.glorot_normal_initializer(),
#                                        name=text+'_2nd_layer',
#                                        reuse=reuse)
#                alph.append(temp2)
#                reuse = True
#            alph = tf.concat(alph, 1, name=text_i)
#            return alph
#        #to call:
#        #alph = alph_fcn(UU, hh, us, ue, 'alph', 'alph_'+str(i),
#        #                scope_decoder_iter.reuse)

        def us_fcn_new(si, UT, text_i):
            # turn si to one-hot. mask UT. reshape
            si_one_hot = tf.one_hot(si, FLAGS.d_max_width+1, True, False,
                                    name=text_i+'one_hot')
            return tf.boolean_mask(UT, si_one_hot, text_i)

        def alph_fcn_1(alph_vars, hh, us, ue, text_i):
            # one hidden layer, size of which is important. need 4*stat_size
            (w12d, w13d, b1, w2, b2, hidden_size, H1Ur, UT,
             d_maski_, _d_maski) = alph_vars

            input2d = tf.concat((hh, us, ue), 1, text_i+'_2d')
            H12d = tf.matmul(input2d, w12d, name=text_i+'_H1d2')
            H12de = tf.expand_dims(H12d, 1, text_i+'_H1d2e')
            H1s = tf.add(tf.add(H1Ur, H12de, text_i+'_H1s1'), b1,
                         text_i+'_H1s')
            H1 = tf.nn.relu(H1s, text_i+'_H1')

            ##residual. requires hidden_size = 7*state_size
            #input2ds = tf.stack([input2d,]*FLAGS.d_max_width, 1,
            #                    text_i+'_2ds')
            #H1 = tf.add(tf.concat((UT, input2ds), 2, text_i+'_inputall'),
            #            H1,
            #            text_i+'_H1_add')

            H1 = tf.nn.dropout(H1, 1-FLAGS.dropout,
                               [tf.shape(H1)[0], 1, tf.shape(H1)[2]],
                               name=text_i+'_H1_d')
            #tf.nn.dropout and tf.layers.dropout are different

            H1r = tf.reshape(H1, (-1, hidden_size), text_i+'_H1r')

            alph = tf.reshape(tf.matmul(H1r, w2, name=text_i+'_H1rw2'),
                              (-1, FLAGS.d_max_width+1),
                              text_i+'_alph_no_b2')
            alph = tf.add(alph, b2, text_i+'_alph_no_mask')
            alph_w_sen = tf.where(tf.is_inf(_d_maski, text_i+'isinf_d_maski'),
                                  _d_maski,
                                  alph,
                                  text_i+'_alph_w_sen')
            alph_wo_sen = tf.where(tf.is_inf(d_maski_, text_i+'isinfd_maski_'),
                                   d_maski_,
                                   alph,
                                   text_i+'_alph_wo_sen')
            return alph_w_sen, alph_wo_sen

        def alph_fcn_1_vars(text, UU, UT, d_maski_, _d_maski):
            hidden_size = (int)(4*FLAGS.state_size) #!!! critical
            w12d = tf.get_variable(text+'_1st_2d_w',
                                   (5*FLAGS.state_size, hidden_size),
                                   tf.float32,
                                   tf.glorot_uniform_initializer())
            #regularizer=tf.contrib.layers.l2_regularizer(FLAGS.alp_reg))
            w13d = tf.get_variable(text+'_1st_3d_w',
                                   (2*FLAGS.state_size, hidden_size),
                                   tf.float32,
                                   tf.glorot_uniform_initializer())
            b1 = tf.get_variable(text+'_1st_b', (hidden_size,), tf.float32,
                                 tf.zeros_initializer())
            # ^ 41592537
            w2 = tf.get_variable(text+'_2nd_w',
                                 (hidden_size, 1),
                                 tf.float32,
                                 tf.glorot_uniform_initializer())
            b2 = tf.get_variable(text+'_2nd_b', (1,), tf.float32,
                                 tf.zeros_initializer())

            UUr = tf.reshape(UT, (-1, 2*FLAGS.state_size), text+'_UUr')
            # ^ UU or UT?
            H1U = tf.matmul(UUr, w13d, name=text+'_H1U')
            H1Ur = tf.reshape(H1U,
                              (-1, FLAGS.d_max_width+1, hidden_size),
                              text+'_H1Ur')
            return (w12d, w13d, b1, w2, b2, hidden_size, H1Ur, UT, d_maski_,
                    _d_maski)

        def alph_fcn_3(alph_vars, hh, us, ue, text_i):
            # maxout
            WD, W1r, b1, UW1Ur, W2, b2, W3, b3, d_maski_, _d_maski = alph_vars

            input2d = tf.concat((hh, us, ue), 1, text_i+'_2d')
            rr = tf.matmul(input2d, WD, name=text_i+'_huuWD')
            rr = tf.tanh(rr, text_i+'_rr')

            rr = tf.nn.dropout(rr, 1-FLAGS.dropout, name=text_i+'_rrd')

            rrW1r = tf.matmul(rr, W1r, name=text_i+'_rrW1r')
            rrW1re = tf.reshape(rrW1r, (-1, 1, FLAGS.state_size, FLAGS.p),
                                text_i+'_rrW1re')
            mt1 = tf.add(tf.add(UW1Ur, rrW1re, text_i+'_UW1Ur_rW1re'), b1,
                         text_i+'_mt1')
            mt1 = tf.reduce_max(mt1, 3, name=text_i+'_mt1r_max')

            mt1 = tf.nn.dropout(mt1, 1-FLAGS.dropout,
                                [tf.shape(mt1)[0], 1, tf.shape(mt1)[2]],
                                name=text_i+'_mt1d')
            # both uses of mt1 are inputs of maxout. so dropout ok

            mt1r = tf.reshape(mt1, (-1, FLAGS.state_size), text_i+'_mt1r')

            mt2r = tf.add(tf.matmul(mt1r, W2, name=text_i+'_mt1r_W2'),
                          b2,
                          text_i+'_mt2r')
            mt2r = tf.reshape(mt2r,
                              (-1, FLAGS.d_max_width+1, FLAGS.state_size,
                               FLAGS.p),
                              text_i+'_mt2rr')
            mt2r = tf.reduce_max(mt2r, 3, name=text_i+'_mt2r_mx')

            mt2r = tf.nn.dropout(mt2r, 1-FLAGS.dropout,
                                 [tf.shape(mt2r)[0], 1, tf.shape(mt2r)[2]],
                                 name=text_i+'_mt2d')

            mt2r = tf.reshape(mt2r, (-1, FLAGS.state_size), text_i+'_mt2r_max')

            mt1mt2 = tf.concat((mt1r, mt2r), 1, text_i+'_mt1mt2')
            # both mt1r and mt2r already have dropout

            alph = tf.add(tf.matmul(mt1mt2, W3, name=text_i+'_mt1mt2W3'),
                          b3,
                          text_i+'_alph_r')
            alph = tf.reshape(alph, (-1, FLAGS.d_max_width+1, FLAGS.p),
                              text_i+'_alph')
            alph = tf.reduce_max(alph, 2, name=text_i+'_alph_max')
            alph_w_sen = tf.where(tf.is_inf(_d_maski, text_i+'isinf_d_maski'),
                                  _d_maski,
                                  alph,
                                  text_i+'_alph_w_sen')
            alph_wo_sen = tf.where(tf.is_inf(d_maski_, text_i+'isinfd_maski_'),
                                   d_maski_,
                                   alph,
                                   text_i+'_alph_wo_sen')
            return alph_w_sen, alph_wo_sen

        def alph_fcn_3_vars(text, UU, UT, d_maski_, _d_maski):
            WD = tf.get_variable(text+'_WD',
                                 (5*FLAGS.state_size, FLAGS.state_size),
                                 tf.float32,
                                 tf.glorot_uniform_initializer())
            bound = np.sqrt(6 / (2*FLAGS.state_size + FLAGS.state_size))
            W1U = tf.get_variable(text+'_W1U',
                                  (2*FLAGS.state_size,
                                   FLAGS.state_size*FLAGS.p),
                                  tf.float32,
                                  tf.random_uniform_initializer(-bound, bound))
            bound = np.sqrt(6 / (FLAGS.state_size + FLAGS.state_size))
            W1r = tf.get_variable(text+'_W1r',
                                  (FLAGS.state_size, FLAGS.state_size*FLAGS.p),
                                  tf.float32,
                                  tf.random_uniform_initializer(-bound, bound))
            b1 = tf.get_variable(text+'_b1',
                                 (FLAGS.state_size, FLAGS.p),
                                 tf.float32,
                                 tf.zeros_initializer())

            UT = tf.nn.dropout(UT, 1-FLAGS.dropout,
                               [tf.shape(UT)[0], 1, tf.shape(UT)[2]],
                               name=text+'_UTd')
            
            UUr = tf.reshape(UT, (-1, 2*FLAGS.state_size), text+'_UUr')
            UW1U = tf.matmul(UUr, W1U, name=text+'_UW1U')
            UW1Ur = tf.reshape(UW1U,
                               (-1, FLAGS.d_max_width+1, FLAGS.state_size,
                                FLAGS.p),
                               text+'_UW1Ur')
            bound = np.sqrt(6 / (FLAGS.state_size + FLAGS.state_size))
            W2 = tf.get_variable(text+'_W2',
                                 (FLAGS.state_size, FLAGS.state_size*FLAGS.p),
                                 tf.float32,
                                 tf.random_uniform_initializer(-bound, bound))
            b2 = tf.get_variable(text+'_b2',
                                 (FLAGS.state_size*FLAGS.p,),
                                 tf.float32,
                                 tf.zeros_initializer())
            bound = np.sqrt(6 / (2*FLAGS.state_size+1))
            W3 = tf.get_variable(text+'_W3',
                                 (2*FLAGS.state_size, FLAGS.p),
                                 tf.float32,
                                 tf.random_uniform_initializer(-bound, bound))
            b3 = tf.get_variable(text+'_b3',
                                 (FLAGS.p,),
                                 tf.float32,
                                 tf.zeros_initializer())
            return WD, W1r, b1, UW1Ur, W2, b2, W3, b3, d_maski_, _d_maski

        si = tf.zeros((tf.shape(UU)[0],), tf.int32, name='si_-1')
        #ei_edge = tf.subtract(d_lens_ph, 1, 'ei_-1') # no obvious help
        ei = si

        cell = tf.nn.rnn_cell.LSTMCell(FLAGS.state_size,
                                       initializer=tf.orthogonal_initializer())

        hh = tf.zeros((tf.shape(UU)[0], FLAGS.state_size), tf.float32, 'hh')
        cc = tf.zeros((tf.shape(UU)[0], FLAGS.state_size), tf.float32, 'cc')
        us = us_fcn_new(si, UT, 'us_-1')
        ue = us_fcn_new(ei, UT, 'ue_-1')

        si_all = []
        ei_all = []
        alph_all = []
        beta_all = []

        alph_function_vars_dict = {}
        alph_function_vars_dict[1] = alph_fcn_1_vars
        #alph_function_vars_dict[2] = alph_fcn_2_vars
        alph_function_vars_dict[3] = alph_fcn_3_vars
        #alph_function_vars_dict[4] = alph_fcn_4_vars
        alph_function_vars = alph_function_vars_dict[FLAGS.alph_fcn]
        alph_function_dict = {}
        alph_function_dict[1] = alph_fcn_1
        #alph_function_dict[2] = alph_fcn_2  # two-layer net. see backup
        alph_function_dict[3] = alph_fcn_3
        #alph_function_dict[4] = alph_fcn_4  # debug for maxout. see backup
        alph_function = alph_function_dict[FLAGS.alph_fcn]
        

        alph_vars = alph_function_vars('alph', UU, UT, d_maski_, _d_maski)
        beta_vars = alph_function_vars('beta', UU, UT, d_maski_, _d_maski)

        with tf.variable_scope('decoder_iter') as scope_decoder_iter:
            pass

        for i in xrange(FLAGS.alph_max_iter):
            usue = tf.concat((us, ue), 1, 'usue'+str(i))
            _, (cc, hh) = cell(usue, (cc, hh), scope_decoder_iter) # 41789133
            # still need scope in a with block

            alph_w_sen, alph_wo_sen = alph_function(alph_vars, hh, us, ue,
                                                    'alph_'+str(i))
            si = tf.argmax(alph_wo_sen, 1, 'si_'+str(i), output_type=tf.int32)
            #si = tf.minimum(si, ei_edge, 'si_m_'+str(i)) # alph already masked
            us = us_fcn_new(si, UT, 'us_'+str(i))

            beta_w_sen, beta_wo_sen = alph_function(beta_vars, hh, us, ue,
                                                    'beta_'+str(i))
            ei = tf.argmax(beta_wo_sen, 1, 'ei_'+str(i), output_type=tf.int32)
            ue = us_fcn_new(ei, UT, 'ue_'+str(i))

            scope_decoder_iter.reuse_variables()

            if FLAGS.suppress_sen_in_alph:
                alph_all.append(alph_wo_sen)
                beta_all.append(beta_wo_sen)
            else:
                alph_all.append(alph_w_sen)
                beta_all.append(beta_w_sen)

            si_all.append(si)
            ei_all.append(ei)

        si_all = tf.stack(si_all, 1, 'si_all')
        ei_all = tf.stack(ei_all, 1, 'ei_all')
        return (alph_all, beta_all, si_all, ei_all)


class QASystem(object):
    def __init__(self, encoder, decoder, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        # ==== set up placeholder tokens ========
        self.d_ph = tf.placeholder(tf.int32,
                                   (None, FLAGS.d_max_width),
                                   'd_ph')
        self.q_ph = tf.placeholder(tf.int32,
                                   (None, FLAGS.q_max_width),
                                   'q_ph')
        self.d_mask_ph = tf.placeholder(tf.bool,
                                        (None, FLAGS.d_max_width),
                                        'd_mask_ph')
        self.q_mask_ph = tf.placeholder(tf.bool,
                                        (None, FLAGS.q_max_width),
                                        'q_mask_ph')
        d_mask_ph_slice = self.d_mask_ph[:, :1]
        #q_mask_ph_slice = self.q_mask_ph[:, :1]
        self._d_mask = tf.concat((d_mask_ph_slice, self.d_mask_ph), 1,
                                 '1_d_mask')
        #self._q_mask = tf.concat((q_mask_ph_slice, self.q_mask_ph), 1,
        #                         '1_q_mask')

        self.d_mask_ = tf.concat((self.d_mask_ph,
                                  tf.zeros_like(d_mask_ph_slice, tf.bool)),
                                 1, 'd_mask_0')
        #self.q_mask_ = tf.concat((self.q_mask_ph,
        #                          tf.zeros_like(q_mask_ph_slice, tf.bool)),
        #                         1, 'q_mask_0')
        d_maski_0 = tf.zeros(tf.shape(self.d_mask_ph), tf.float32,
                             name='d_maski_0s')
        q_maski_0 = tf.zeros(tf.shape(self.q_mask_ph), tf.float32,
                             name='q_maski_0s')
        fals = tf.constant(False, tf.bool, (), 'fals')
        ninf = tf.constant(-np.inf, tf.float32, (), 'ninf')
        d_maski_i = tf.add(d_maski_0, ninf, 'd_maski_i_add')
        q_maski_i = tf.add(q_maski_0, ninf, 'q_maski_i_add')
        d_mask_eq = tf.equal(self.d_mask_ph, fals, 'd_maski_eq')
        q_mask_eq = tf.equal(self.q_mask_ph, fals, 'q_maski_eq')
        self.d_maski = tf.where(d_mask_eq, d_maski_i, d_maski_0, 'd_maski')
        self.q_maski = tf.where(q_mask_eq, q_maski_i, q_maski_0, 'q_maski')
        _d_maski = tf.concat((d_maski_0[:, :1], self.d_maski), 1, '0_d_maski')
        _q_maski = tf.concat((q_maski_0[:, :1], self.q_maski), 1, '0_q_maski')
        d_maski_ = tf.concat((self.d_maski, d_maski_i[:, :1]), 1, 'd_maski_')
        #q_maski_ = tf.concat((self.q_maski, q_maski_i[:, :1]), 1, 'q_maski_')
        self._d_maski = _d_maski
        #self._q_maski = _q_maski
        self.d_maski_ = d_maski_
        #self.q_maski_ = q_maski_

        #dq_maski_head = tf.zeros((tf.shape(self.d_maski)[0], 1),
        #                         tf.float32,
        #                         'dq_maski_head')
        #d_maskih = tf.concat((d_maski_0[:, :1], self.d_maski), 1, 'd_maskih')
        #q_maskih = tf.concat((q_maski_0[:, :1], self.q_maski), 1, 'q_maskih')

        self.d_maskih = tf.stack([_d_maski,]*(FLAGS.q_max_width+1), 1,
                                 'd_maskih')
        self.q_maskih = tf.stack([_q_maski,]*(FLAGS.d_max_width+1), 1,
                                 'q_maskih')
    
        self.span_ph = tf.placeholder(tf.int32, (None, 2), 'span_ph')
        self.d_lens_ph = tf.placeholder(tf.int32, (None,), 'd_lens_ph')
        self.q_lens_ph = tf.placeholder(tf.int32, (None,), 'q_lens_ph')
        one = tf.constant(1, tf.int32, (), 'one')
        self.d_lens_1 = tf.add(self.d_lens_ph, one, 'd_lens_1')

        d_lens_one_hot = tf.one_hot(self.d_lens_ph,
                                    FLAGS.d_max_width+1,
                                    dtype=tf.int32,
                                    name='d_lens_one_hot')
        q_lens_one_hot = tf.one_hot(self.q_lens_ph,
                                    FLAGS.q_max_width+1,
                                    dtype=tf.int32,
                                    name='q_lens_one_hot')

        dq_sen_0 = tf.zeros((1, FLAGS.state_size), name='dq_sen_0')
        d_sen_v = tf.get_variable('d_sen_v',
                                  (1, FLAGS.state_size),
                                  tf.float32,
                                  tf.glorot_uniform_initializer())
        q_sen_v = tf.get_variable('q_sen_v',
                                  (1, FLAGS.state_size),
                                  tf.float32,
                                  tf.glorot_uniform_initializer())
        d_sen_2 = tf.concat((dq_sen_0, d_sen_v), 0, 'd_sen_2')
        q_sen_2 = tf.concat((dq_sen_0, q_sen_v), 0, 'q_sen_2')
        self.d_sen_m = tf.nn.embedding_lookup(d_sen_2,
                                              d_lens_one_hot,
                                              name='d_sen_m')
        self.q_sen_m = tf.nn.embedding_lookup(q_sen_2,
                                              q_lens_one_hot,
                                              name='q_sen_m')

        d_sen_mask = tf.stack([d_lens_one_hot]*FLAGS.state_size, 2, 'dsenmask')
        q_sen_mask = tf.stack([q_lens_one_hot]*FLAGS.state_size, 2, 'qsenmask')

        #one = tf.constant(1, tf.int32, (), 'one')
        self.d_sen_mask_eq = tf.equal(d_sen_mask, one, 'd_sen_mask_eq')
        self.q_sen_mask_eq = tf.equal(q_sen_mask, one, 'q_sen_mask_eq')

        self.embed_path = args[0]
        self.rev_vocab = args[1]
        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system(encoder, decoder)
            self.setup_loss()

        # ==== set up training/updating procedure ====
        self.saver = tf.train.Saver()  # this has to come after all vars def

    def setup_system(self, encoder, decoder):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        tt = time.time()  
        UT = encoder.encode(
                [self.d_emb, self.q_emb],
                [self.d_maskih, self.q_maskih,
                 self.d_maski_,
                 self._d_maski],
                [self.d_lens_ph, self.q_lens_ph, self.d_lens_1],
                [self.d_sen_m, self.q_sen_m],
                [self.d_sen_mask_eq, self.q_sen_mask_eq],
                None)
        print(time.time()-tt, 'in encoder.encode')

        tt = time.time()  
        (self.alph_all, self.beta_all,
         self.si_all, self.ei_all) = decoder.decode(UT)

        print(time.time()-tt, 'in decoder.decode')
        #raise NotImplementedError("Connect all parts of your system here!")

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        #with vs.variable_scope("loss"):
        tt = time.time()
        '''
        # choice 1: choose among all words in doc
        labels0 = tf.stack([self.span_ph[:, 0]] *FLAGS.alph_max_iter,
                           1, 'span0stack')
        labels1 = tf.stack([self.span_ph[:, 1]] *FLAGS.alph_max_iter,
                           1, 'span1stack')
        alph_all = tf.stack(self.alph_all, 1, 'alph_all')
        beta_all = tf.stack(self.beta_all, 1, 'beta_all')

        loss_alph = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=labels0,
                            logits=alph_all,
                            name='loss_alph'),
                        name='loss_alph_mean')
        loss_beta = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=labels1,
                            logits=beta_all,
                            name='loss_beta'),
                        name='loss_beta_mean')
        self.loss = tf.add(loss_alph, loss_beta, 'loss')
        '''
        # choice 2: for each word in doc, is it chosen or not?
        alph_gold = tf.one_hot(self.span_ph[:, 0],
                               FLAGS.d_max_width+1,
                               dtype=tf.float32,
                               name='alph_gold')
        beta_gold = tf.one_hot(self.span_ph[:, 1],
                               FLAGS.d_max_width+1,
                               dtype=tf.float32,
                               name='beta_gold')
        self.loss = tf.zeros((), tf.float32, 'loss')

        if FLAGS.suppress_sen_in_alph:
            d_mask_to_use = self.d_mask_
        else:
            d_mask_to_use = self._d_mask

        for i in xrange(FLAGS.alph_max_iter):
            loss_alph = tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=alph_gold,
                            logits=self.alph_all[i],
                            name='loss_alph_'+str(i))
            loss_beta = tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=beta_gold,
                            logits=self.beta_all[i],
                            name='loss_beta_'+str(i))
            loss_alph = tf.boolean_mask(loss_alph,
                                        d_mask_to_use,
                                        'loss_alph_masked_'+str(i))
            loss_beta = tf.boolean_mask(loss_beta,
                                        d_mask_to_use,
                                        'loss_beta_masked_'+str(i))
            loss_alph = tf.reduce_mean(loss_alph,
                                       name='loss_alph_mean_'+str(i))
            loss_beta = tf.reduce_mean(loss_beta,
                                       name='loss_beta_mean_'+str(i))
            loss = tf.add(loss_alph, loss_beta, 'loss_mean_'+str(i))
            self.loss += tf.add(self.loss, loss, 'loss_add_'+str(i))


        #self.train_op = tf.train.AdamOptimizer(
        #        learning_rate=FLAGS.learning_rate).minimize(self.loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        grad, var = zip(*optimizer.compute_gradients(self.loss))

        grad, _ = tf.clip_by_global_norm(grad, FLAGS.max_gradient_norm)
            # ^ this is norm before clipping

        self.grad_norm = tf.global_norm(grad)
        if FLAGS.ifgpu:
            self.mem = tf.contrib.memory_stats.MaxBytesInUse()  # 40190510
        else:
            self.mem = self.grad_norm # if there is no gpu..

        self.train_op = optimizer.apply_gradients(zip(grad, var))

        self.grad = grad
        self.var = var

        print(time.time()-tt, 'in setup_loss')

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        glove = np.load(self.embed_path)
        glove = glove[glove.keys()[0]]
        glove = np.float32(glove)
        #with vs.variable_scope("embeddings"):
        embeddings = tf.get_variable('embeddings',
                                     initializer=glove,
                                     trainable=False)

        self.q_emb = tf.nn.embedding_lookup(
                embeddings, self.q_ph, name='q_emb')
        self.d_emb = tf.nn.embedding_lookup(
                embeddings, self.d_ph, name='d_emb')

#    def optimize(self, session, train_x, train_y):
#        """
#        Takes in actual data to optimize your model
#        This method is equivalent to a step() function
#        :return:
#        """
#        input_feed = {}
#
#        # fill in this feed_dictionary like:
#        # input_feed['train_x'] = train_x
#
#        output_feed = []
#
#        outputs = session.run(output_feed, input_feed)
#
#        return outputs
#
#    def test(self, session, valid_x, valid_y):
#        """
#        in here you should compute a cost for your validation set
#        and tune your hyperparameters according to the validation set performance
#        :return:
#        """
#        input_feed = {}
#
#        # fill in this feed_dictionary like:
#        # input_feed['valid_x'] = valid_x
#
#        output_feed = []
#
#        outputs = session.run(output_feed, input_feed)
#
#        return outputs
#
#    def decode(self, session, test_x):
#        """
#        Returns the probability distribution over different positions in the paragraph
#        so that other methods like self.answer() will be able to work properly
#        :return:
#        """
#        input_feed = {}
#
#        # fill in this feed_dictionary like:
#        # input_feed['test_x'] = test_x
#
#        output_feed = []
#
#        outputs = session.run(output_feed, input_feed)
#
#        return outputs
#
#    def answer(self, session, test_x):
#
#        yp, yp2 = self.decode(session, test_x)
#
#        a_s = np.argmax(yp, axis=1)
#        a_e = np.argmax(yp2, axis=1)
#
#        return (a_s, a_e)
#
#    def validate(self, sess, valid_dataset):
#        """
#        Iterate through the validation dataset and determine what
#        the validation cost is.
#
#        This method calls self.test() which explicitly calculates validation cost.
#
#        How you implement this function is dependent on how you design
#        your data iteration function
#
#        :return:
#        """
#        valid_cost = 0
#
#        for valid_x, valid_y in valid_dataset:
#          valid_cost = self.test(sess, valid_x, valid_y)
#
#
#        return valid_cost
#
#    def evaluate_answer(self, session, dataset, sample=100, log=False):
#        """
#        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
#        with the set of true answer labels
#
#        This step actually takes quite some time. So we can only sample 100 examples
#        from either training or testing set.
#
#        :param session: session should always be centrally managed in train.py
#        :param dataset: a representation of our data, in some implementations, you can
#                        pass in multiple components (arguments) of one dataset to this function
#        :param sample: how many examples in dataset we look at
#        :param log: whether we print to std out stream
#        :return:
#        """
#
#        f1 = 0.
#        em = 0.
#
#        if log:
#            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))
#
#        return f1, em

    def create_feed_dict(self,
                         doc, qst,
                         d_mask, q_mask,
                         d_lens, q_lens,
                         span=None):
        feed = {}
        feed[self.d_ph] = doc
        feed[self.q_ph] = qst
        feed[self.d_mask_ph] = d_mask
        feed[self.q_mask_ph] = q_mask
        feed[self.d_lens_ph] = d_lens
        feed[self.q_lens_ph] = q_lens
        if span is not None:
            feed[self.span_ph] = span
        return feed

    def get_mini(self, dataset_trainOrVal, batch_size, shuffle, span=True):
        feed = [dataset_trainOrVal['.ids.context'],
                dataset_trainOrVal['.ids.question'],
                dataset_trainOrVal['.ids.context.mask'],
                dataset_trainOrVal['.ids.question.mask'],
                dataset_trainOrVal['.ids.context.lens'],
                dataset_trainOrVal['.ids.question.lens']]
        if span:
            feed.append(dataset_trainOrVal['.span'])
        return util.get_minibatches(feed, batch_size, shuffle)

    def _run_epoch_train_minibatch(self, session, train_x_and_y):
        '''
        trains one minibatch
        '''
        feed = self.create_feed_dict(*train_x_and_y)
        fetch = [self.train_op, self.loss, self.grad_norm, self.mem]#
                 #self.grad, self.var]
        return session.run(fetch, feed)

    def _run_epoch_val_minibatch(self, sess, val_x_and_y):
        '''
        validates one minibatch
        '''
        feed = self.create_feed_dict(*val_x_and_y)
        fetch = [self.si_all, self.ei_all, self.loss]
        return sess.run(fetch, feed)

    def _run_epoch_dev_minibatch(self, sess, dev_x):
        '''
        test ("dev") one minibatch
        '''
        feed = self.create_feed_dict(*dev_x)
        fetch = [self.si_all, self.ei_all, self.mem]
        return sess.run(fetch, feed)

    def _run_epoch_train_part(self, sess, dataset_train):
        '''
        trains for one epoch
        '''
        prog = util.Progbar(
                target=1 + int(len(dataset_train['.ids.context']) / \
                               FLAGS.batch_size))
        for i, batch in enumerate(self.get_mini(dataset_train,
                                                FLAGS.batch_size,
                                                shuffle=True,
                                                span=True)):
            out = self._run_epoch_train_minibatch(sess, batch)
            #_, loss, gnorm, mem, grad, var = out
            _, loss, gnorm, mem = out
            mem = int(mem)>>20
            prog.update(i + 1, exact=[('train loss', loss),
                                      ('gnorm', gnorm),
                                      ('mem', mem)])  # mb
            if np.isnan(gnorm):
                logging.info('gnorm nan')
                #np.save('nan_grad', grad)
                #np.save('nan_var', var)
                raise Exception('gnorm nan')

        logging.info(
            'last train loss {}, gnorm {}, mem {}'.format(loss, gnorm, mem))

    def _run_epoch_val_part(self, sess, dataset_val):
        '''
        validates for one epoch
        returns validation metric
        which is the harmonic mean of f1 and em
        '''
        si_pred = []
        ei_pred = []
        #losses = []
        prog = util.Progbar(
                target=1 + int(len(dataset_val['.ids.context']) / \
                               FLAGS.batch_size))
        for i, batch in enumerate(self.get_mini(dataset_val,
                                                FLAGS.batch_size,
                                                shuffle=False,
                                                span=True)):

            si_all, ei_all, loss = self._run_epoch_val_minibatch(sess, batch)
            prog.update(i + 1, exact=[('val loss', loss)])
            si_pred.append(si_all)  # order should be preserved. shuffle=False
            ei_pred.append(ei_all)
            #losses.append(loss)

        si_pred = np.concatenate(si_pred)
        ei_pred = np.concatenate(ei_pred)
        span = self.pick_si_ei(si_pred, ei_pred)
        answers = self.span_to_pred(span, dataset_val['.ids.context'])
        f1, em = self.evaluate(answers, dataset_val['.answer'])
        #f1f1em = 2/(1/f1+1/em)
        f1f1em = 2/(1/max(f1, 1e-10)+1/max(em, 1e-10))  # for small datasets
        logging.info('last val loss {}'.format(loss))
        logging.info("F1: {}, EM: {}, F1F1EM: {}".format(f1, em, f1f1em))
        return f1f1em
        #return em
        #return sum(losses)/len(losses)  # was used as figure of merit

    def _run_epoch_dev_part(self, sess, dataset_dev):
        '''
        test ("dev") for one epoch
        returns predicted answers
        might be able to merge with _run_epoch_val_part. didn't bother
        '''
        si_pred = []
        ei_pred = []
        prog = util.Progbar(
                target=1 + int(len(dataset_dev['.ids.context']) / \
                               FLAGS.batch_size))
        for i, batch in enumerate(self.get_mini(dataset_dev,
                                                FLAGS.batch_size,
                                                shuffle=False,
                                                span=False)):

            si_all, ei_all, mem = self._run_epoch_dev_minibatch(sess, batch)
            mem = int(mem)>>20
            prog.update(i + 1, exact=[('mem', mem)])
            si_pred.append(si_all)  # order should be preserved. shuffle=False
            ei_pred.append(ei_all)

        si_pred = np.concatenate(si_pred)
        ei_pred = np.concatenate(ei_pred)
        span = self.pick_si_ei(si_pred, ei_pred)
        answers = self.span_to_pred(span, dataset_dev['.ids.context'])
        return answers

    def evaluate(self, answers, gold):
        '''
        calculates f1 and em, given a batch of guesses and gold data
        '''
        num = len(answers)        
        assert num == len(gold)
        f1 = 0.
        em = 0.
        for i in xrange(num):
            f1 += f1_score(answers[i], gold[i])
            emm = exact_match_score(answers[i], gold[i])
            em += emm
            #print(i, str(emm)[0], '|', answers[i], '|', gold[i])
        return (f1/num, em/num)

    def span_to_pred(self, span, d_tokens):
        '''
        returns predicted strings
        span: (None, 2), the start and the end indices
        d_tokens: document in the form of tokens
        '''
        def reverse_lookup(i):
            return self.rev_vocab[i]
        out = []
        for i in xrange(len(span)):
            temp = d_tokens[i, span[i, 0]:span[i, 1]+1]
            temp = map(reverse_lookup, temp)
            temp = ' '.join(temp)
            out.append(temp)
        return out

    def pick_si_ei(self, si_all, ei_all):
        '''
        si_all and ei_all are (None, 4)
        pick according to paper
        
        is it possible to do this with tf?
        '''
        numSamples = len(si_all)
        span = np.zeros((numSamples, 2), int)
        m1 = FLAGS.alph_max_iter-1
        for i in xrange(numSamples):
            span[i, 0] = si_all[i, m1]  # default last one
            span[i, 1] = ei_all[i, m1]
            for j in xrange(1, m1):
                j1 = j-1
                if si_all[i, j] == si_all[i, j1] and \
                   ei_all[i, j] == ei_all[i, j1]:
                    span[i, 0] = si_all[i, j]
                    span[i, 1] = ei_all[i, j]
                    break
            if span[i, 0] > span[i, 1]:  # sometimes true..
                span[i] = (span[i, 1], span[i, 0])
        return span
                    
    def run_epoch(self, sess, dataset):
        '''
        trains and validates for one epoch
        returns validation metric, harmonic mean of f1 and em
        '''
        self._run_epoch_train_part(sess, dataset['train'])
        val = self._run_epoch_val_part(sess, dataset['val'])
        return val
        
    def train(self, session, dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training
        '''
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
        '''

        best_score = 0.
        for epoch in range(FLAGS.epochs):
            logging.info('Epoch %d out of %d', epoch + 1, FLAGS.epochs)
            score = self.run_epoch(session, dataset)
            if score > best_score:
                best_score = score
                logging.info('New best score! Saving model in %s', train_dir)
                self.saver.save(session, pjoin(train_dir,
                                               FLAGS.save_file_name))
        return best_score


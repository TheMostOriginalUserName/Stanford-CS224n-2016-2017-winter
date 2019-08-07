from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf
#
tf.reset_default_graph()
from tensorflow.python.platform import gfile
#import matplotlib.pyplot as pl
import numpy as np
import sys ##### 67631. to run in parent dir but include qa_model
sys.path.append('./code/')
import argparse ##### tf issues 8389. to reset flags
tf.flags._global_parser = argparse.ArgumentParser()
import time
#from tensorflow.python import debug as tfdbg
from tensorflow.python.client import device_lib  # 38559755
#
from qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 80, "Batch size to use during training.")  # was 10
tf.app.flags.DEFINE_integer("epochs", 30, "Number of epochs to train.")  # was 10
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 300, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
#tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
#tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
#tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
#tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")

FLAGS = tf.app.flags.FLAGS

#
tf.app.flags.DEFINE_integer('d_max_width', 600, '')  # should use 600. 100 for speed
tf.app.flags.DEFINE_integer('q_max_width', 60, '')  # should use 60. 30 for speed
tf.app.flags.DEFINE_integer('alph_max_iter', 4, '')  # should use 4. 2 for speed
#tf.app.flags.DEFINE_integer('q_reg', 1e-4, '')
#tf.app.flags.DEFINE_integer('alp_reg', 1e-4, '')
tf.app.flags.DEFINE_integer('p', 4, '')

tf.app.flags.DEFINE_string('run_name', '', '')
tf.app.flags.DEFINE_string('log_dir', 'log'+FLAGS.run_name, '')
tf.app.flags.DEFINE_string('save_file_name', 'best'+FLAGS.run_name, '')
#tf.app.flags.DEFINE_string('train_dir', 'train'+FLAGS.run_name, '')
#^ can't specify save dir easily. each run rewrites symlink in /tmp and uses it
tf.app.flags.DEFINE_integer('data_reduction', 1, '')
tf.app.flags.DEFINE_integer('alph_fcn', 3, '')
tf.app.flags.DEFINE_bool('suppress_sen_in_U', True, '')
tf.app.flags.DEFINE_bool('suppress_sen_in_alph', False, '')
tf.app.flags.DEFINE_bool('ifgpu', False, '')
# ^ will be overwritten if gpu is present. can't initialize this in main()

'''
Notes
- LSTM is slightly easier to train than GRU here.
- sequence_length in dynamic_rnn is crucial here.
- To mask softmax:
    Feed -inf with where() instead of +.
    Remove resultant nan using the same trick.
- reshape() starts from the last dim.
- saver.save() needs a file name, not a dir name.
- 840B glove is better than 6B here.
- Avoid tf.while and for loops. See old implementations of us_fcn and alph_fcn.
- The randomness of initialization and minibatches, even if set correctly,
  matters a lot. See training strategies below for example.
- What to do if loss jumps around after several epochs in training?
    Train with a small data set to check if model can overfit.
    Check for nan and inf by tfdbg, fetching them, or saving them.
    Anneal the learning rate.
    (adam can be confused if the starting learning rate is way off.)

Handling of sentinel vecs (sen):

- The influence of sen is blended into UU via CQ. See encode().
- sen can also be allowed to have direct influence on UU's (d_lens_ph+1)th vec.
  (Call that the "tail vec.")
  If suppress_sen_in_U==True/False, tail vec is 0/influeced by sen.
- Tail vec's influence on loss can be toggled. See decode() and setup_loss().
  If suppress_sen_in_alph==True/False,
  the classifier won't/will consider the logit influenced by tail vec.

Training strategy used for alph_fcn==1:
Step 1 to reach >45% EM val
 - learning_rate        .01
 - suppress_sen_in_U    True
 - suppress_sen_in_alph False
Step 2 to reach >50% EM val
 - learning_rate        .01
 - suppress_sen_in_U    True
 - suppress_sen_in_alph True

Step 1 is affected by the randomness of initialization and minibatches.
A good run can reach       >10% EM after one epoch(s)
                           >40%          two
  bad          hang around <20%          ten
Other combinations of flags for sen (except True, False) were always bad.
Lowering or annealing the learning rate might help. But it's interesting that
allowing zero vecs in the classifier can speed up training, and suppressing
zero vecs at step 2 led to improvement.

Training strategy used for alph_fcn==3:
Step 1
 - learning_rate        .001
 - suppress_sen_in_U    True
 - suppress_sen_in_alph False
Step 2
 - learning_rate        .0005
 - suppress_sen_in_U    True
 - suppress_sen_in_alph False
Further annealing can likely help.

batch_size
          alph_fcn
card     1         3
k80      160-180   160
p100     240       220
'''

def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir


def line2nums(line, maxLength, outputArray):
    '''
    line: a string of numbers separated by spaces
    maxLength: the max number of numbers to be recorded in outputArray
    outputArray: as above
    returns oldLen, the number of numbres
    '''
    nums = line.split()
    oldLen = len(nums)
    nums = nums[:maxLength]
    nums = map(int, nums)  # 7368789
    outputArray[:len(nums)] = nums
    return oldLen


def len2mask(lens, max_width):
    '''
    lens: list of numbers. lengths of questions or documents
    returns out
    each row is a mask
    '''
    out1 = np.zeros((len(lens), max_width), dtype=bool)
    for i in xrange(len(lens)):
        lll = min(lens[i], max_width)  # redundant if newLens
        out1[i, :lll] = [True] * lll
    return out1


def oldLensNewLens(oldLens, max_width):
    def fun(x):
        return min(x, max_width)
    return map(fun, oldLens)


## Old version
#def load_data(path,
#              d_max_width=FLAGS.d_max_width,
#              q_max_width=FLAGS.q_max_width):
#    '''
#    returns a nested dict[A][B]
#    A can be 'train', 'val'
#    B can be
#                ['.ids.context',
#                 '.ids.question',
#                 '.span',
#                 '.ids.context.mask', # True/False mask
#                 '.ids.question.mask',
#                 #'.ids.context.maski', # 0./-np.inf mask
#                 #'.ids.question.maski',
#                 '.ids.context.lens',
#                 '.ids.question.lens',
#                 '.answer']
#    each '.len' element is a list of integers
#    each '.answer' element is a list of strings
#    others are np ndarrays
#    '''
#    tt = time.time()
#    names1 = ['train', 'val']
#    names2 = ['.ids.context',
#              '.ids.question',
#              '.span']
#    widths = dict(zip(names2, [d_max_width, q_max_width, 2]))
#
#    def num_of_lines(filepath):
#        with gfile.GFile(filepath, mode='r') as tokens_file:
#            string = tokens_file.read()
#        return len(string.split('\n'))
#
#    REDUCTION = FLAGS.data_reduction
#
#    numOfLines = {}
#    for name1 in names1:
#        numOfLines[name1] = num_of_lines(path+'/'+name1+names2[2]) // REDUCTION
#
#    outDict = {}
#    for name1 in names1:
#        outDict[name1] = {}
#        for i, name2 in enumerate(names2):
#            with gfile.GFile(path+'/'+name1+name2, mode='r') as tokens_file:
#                string = tokens_file.read()
#            oldLens = []
#            manyLines = np.zeros((numOfLines[name1], widths[name2]), dtype=int)
#            for j, line in enumerate(string.split('\n')):
#
#                if j >= numOfLines[name1]:
#                    break
#
#                # read a line and count its length
#                oldLen = line2nums(line, widths[name2], manyLines[j, :])
#                oldLens.append(oldLen)
#            outDict[name1][name2] = manyLines
#            if i < 2:
#                newLens = oldLensNewLens(oldLens, widths[name2])
#                outDict[name1][name2+'.lens'] = newLens
#                out1 = len2mask(oldLens, widths[name2])
#                outDict[name1][name2+'.mask'] = out1
#            print('done reading '+name1+name2)
#
#        name2 = '.answer'
#        with gfile.GFile(path+'/'+name1+name2, mode='r') as tokens_file:
#            string = tokens_file.read()
#        outDict[name1][name2] = string.split('\n')[:numOfLines[name1]]
#        print('done reading '+name1+name2)
#
#    '''
#    def plotcdf(nums):
#        pl.figure()
#        sorted_ = np.sort(nums)
#        yvals = np.arange(len(sorted_))/float(len(sorted_))
#        pl.plot(sorted_, yvals)
#    # is newLens or oldLens recorded?
#    plotcdf(outDict['train']['.ids.context.lens'])
#    plotcdf(outDict['val']  ['.ids.context.lens'])
#    plotcdf(outDict['train']['.ids.question.lens'])
#    plotcdf(outDict['val']  ['.ids.question.lens'])
#    pl.show()
#    raise Exception('')
#    '''
#
#    def trim(dat, length=FLAGS.d_max_width):
#        '''
#        check if dat['.span'] is beyond max_width
#        if yes, remove the sample
#        or else might encounter error when generating
#        one-hot truth labels
#        '''
#        names = dat.keys()
#        indices = []
#        for i in xrange(len(dat['.span'])):
#            if dat['.span'][i, 0] >= length or\
#               dat['.span'][i, 1] >= length or\
#               dat['.ids.context.lens'][i] <= 0:
#                indices.append(i)
#        for name in names:
#            dat[name] = np.delete(dat[name], indices, axis=0)
#
#    trim(outDict['train'])
#    trim(outDict['val'])
#    
#    print(time.time()-tt, 'in load_data')
#    return outDict


def num_of_lines(filepath):
    with gfile.GFile(filepath, mode='r') as tokens_file:
        string = tokens_file.read()
    return len(string.split('\n'))


def trim(dat, length=FLAGS.d_max_width):
    '''
    check if dat['.span'] is beyond max_width
    if yes, remove the sample
    or else might encounter error when generating
    one-hot truth labels
    '''
    names = dat.keys()
    indices = []
    for i in xrange(len(dat['.span'])):
        if dat['.span'][i, 0] >= length or\
           dat['.span'][i, 1] >= length or\
           dat['.ids.context.lens'][i] <= 0 or\
           dat['.ids.question.lens'][i] <= 0:
            indices.append(i)
    for name in names:
        dat[name] = np.delete(dat[name], indices, axis=0)


def trim_empty(dat, length=FLAGS.d_max_width):
    '''
    check for empty entries
    if empty, remove the sample
    '''
    names = dat.keys()
    indices = []
    for i in xrange(len(dat[names[0]])):
        if dat['.ids.context.lens'][i] <= 0 or\
           dat['.ids.question.lens'][i] <= 0:
            indices.append(i)
    for name in names:
        dat[name] = np.delete(dat[name], indices, axis=0)
    return indices


# New version. Allows qa_answer.py to call
def load_data_dq(outDict,
                 key,
                 path,
                 d_max_width=FLAGS.d_max_width,
                 q_max_width=FLAGS.q_max_width):
    '''
    reads *.ids.context and *.ids.question files
    returns data through outDict
    creates entries: outDict[key][B]
    key can be 'train', 'val', 'dev'
    B can be
                ['.ids.context',
                 '.ids.question',
                 '.ids.context.mask', # True/False mask
                 '.ids.question.mask',
                 '.ids.context.lens',
                 '.ids.question.lens']
    each '.len' element is a list of integers
    others are np ndarrays
    '''
    tt = time.time()

    names2 = ['.ids.context', '.ids.question']
    widths = [d_max_width, q_max_width]

    numOfLines  = num_of_lines(path+'/'+key+names2[0]) 
    numOfLines1 = num_of_lines(path+'/'+key+names2[1])
    assert numOfLines == numOfLines1
    numOfLinesR = numOfLines // FLAGS.data_reduction

    if not (key in outDict.keys()):
        outDict[key] = {}

    for i, name2 in enumerate(names2):
        with gfile.GFile(path+'/'+key+name2, mode='r') as tokens_file:
            string = tokens_file.read()
        stringsplit = string.split('\n')
        assert len(stringsplit) == numOfLines
        manyLines = np.zeros((numOfLinesR, widths[i]), dtype=int)
        oldLens = []
        for j, line in enumerate(stringsplit):
            if j >= numOfLinesR:
                break
            # read a line and count its length
            oldLen = line2nums(line, widths[i], manyLines[j, :])
            oldLens.append(oldLen)

        outDict[key][name2] = manyLines
        newLens = oldLensNewLens(oldLens, widths[i])
        outDict[key][name2+'.lens'] = newLens
        out1 = len2mask(oldLens, widths[i])
        outDict[key][name2+'.mask'] = out1
        print('done reading '+key+name2)
    print(time.time()-tt, 'in load_data_dq', key)
    return numOfLines


def load_data_sa(outDict, key, path, numOfLinesDQ):
    '''
    reads *.span and *.answer files
    returns data through outDict
    creates entries:
        outDict[key]['.span']   : np ndarray
        outDict[key]['.answer'] : list of strings
    key can be 'train', 'val'
    '''
    tt = time.time()

    name2 = '.span'

    numOfLines = num_of_lines(path+'/'+key+name2)
    assert numOfLines == numOfLinesDQ
    numOfLinesR = numOfLines // FLAGS.data_reduction

    if not (key in outDict.keys()):
        outDict[key] = {}

    with gfile.GFile(path+'/'+key+name2, mode='r') as tokens_file:
        string = tokens_file.read()
    stringsplit = string.split('\n')
    assert len(stringsplit) == numOfLines
    manyLines = np.zeros((numOfLinesR, 2), dtype=int)
    for j, line in enumerate(stringsplit):
        if j >= numOfLinesR:
            break
        # read a line and count its length
        oldLen = line2nums(line, 2, manyLines[j, :])
        if oldLen != 2:  # Does happen. Taken care of by trim()
            pass#raise Exception(oldLen)
    outDict[key][name2] = manyLines
    print('done reading '+key+name2)

    name2 = '.answer'

    numOfLines = num_of_lines(path+'/'+key+name2)
    assert numOfLines == numOfLinesDQ
    numOfLinesR = numOfLines // FLAGS.data_reduction

    with gfile.GFile(path+'/'+key+name2, mode='r') as tokens_file:
        string = tokens_file.read()
    stringsplit = string.split('\n')
    assert len(stringsplit) == numOfLines
    outDict[key][name2] = stringsplit[:numOfLinesR]
    print('done reading '+key+name2)
    print(time.time()-tt, 'in load_data_sa', key)


def main(_):

    # Do what you need to load datasets from FLAGS.data_dir
    #dataset = load_data(FLAGS.data_dir)  # None
    dataset = {}
    num_train = load_data_dq(dataset, 'train', FLAGS.data_dir)
    num_val   = load_data_dq(dataset, 'val'  , FLAGS.data_dir)
    load_data_sa(dataset, 'train', FLAGS.data_dir, num_train)
    load_data_sa(dataset, 'val'  , FLAGS.data_dir, num_val)
    trim(dataset['train'])
    trim(dataset['val'])

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)

    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
    decoder = Decoder(output_size=FLAGS.output_size)

    # Session moved upfront to set the ifgpu flag before QASystem
    with tf.Session() as sess:
        pass
    local_device_protos = device_lib.list_local_devices()  # 38559755
    for x in local_device_protos:
        if x.device_type == 'GPU':
            FLAGS.ifgpu = True
            break

    qa = QASystem(encoder, decoder, embed_path, rev_vocab)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    #print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    # tfdbg
    #with tf.Session() as sess:
    #    pass
    #sess = tfdbg.LocalCLIDebugWrapperSession(sess)
    #sess.add_tensor_filter("has_inf_or_nan", tfdbg.has_inf_or_nan)

    load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
    initialize_model(sess, qa, load_train_dir)

    save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
    qa.train(sess, dataset, save_train_dir)

    #qa.evaluate_answer(sess, dataset, vocab, FLAGS.evaluate, log=True)
    sess.close()#tfdbg
if __name__ == "__main__":
    tf.app.run()


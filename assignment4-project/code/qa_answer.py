from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import json
import sys
import random
from os.path import join as pjoin

from tqdm import tqdm
import numpy as np
from six.moves import xrange
import tensorflow as tf
#
tf.reset_default_graph()
sys.path.append('./code/')
import argparse ##### tf issues 8389. to reset flags
from train import line2nums, len2mask, oldLensNewLens
tf.flags._global_parser = argparse.ArgumentParser()
# See reload(sys) inside squad_preprocess.py.
# It redirects all print output to std instead of in spyder.
from tensorflow.python.client import device_lib  # 38559755
from preprocessing.squad_preprocess import read_write_dataset
from train import load_data_dq, trim_empty
import time
from tensorflow.python.platform import gfile
#
from qa_model import Encoder, QASystem, Decoder
from preprocessing.squad_preprocess import data_from_json, maybe_download, squad_base_url, \
    invert_map, tokenize, token_idx_map
import qa_data

import logging

logging.basicConfig(level=logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")  # was 0.001
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 180, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 0, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size", 300, "Size of the pretrained vocabulary.")  # was 100
tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_string("dev_path", "data/squad/dev-v1.1.json", "Path to the JSON dev set to evaluate against (default: ./data/squad/dev-v1.1.json)")
#
tf.app.flags.DEFINE_integer('d_max_width', 600, 'what')  # should use 600. 100 for speed
tf.app.flags.DEFINE_integer('q_max_width', 60, 'what')  # should use 60. 30 for speed
tf.app.flags.DEFINE_integer('alpha_max_iter', 4, 'what')  # should use 4. 2 for speed
#tf.app.flags.DEFINE_integer('q_reg', 1e-4, 'what')
#tf.app.flags.DEFINE_integer('alp_reg', 1e-4, 'what')
tf.app.flags.DEFINE_integer('p', 4, 'what')

tf.app.flags.DEFINE_integer('data_reduction', 1, '')
tf.app.flags.DEFINE_integer('alph_fcn', 3, '')
tf.app.flags.DEFINE_bool('suppress_sen_in_U', True, '')
tf.app.flags.DEFINE_bool('suppress_sen_in_alph', True, '')
tf.app.flags.DEFINE_bool('ifgpu', False, '')
# ^ Will be overwritten if gpu is present. Can't initialize this in main().
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
#

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


def read_dataset(dataset, tier, vocab):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""

    context_data = []
    query_data = []
    question_uuid_data = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']
            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)

            qas = article_paragraphs[pid]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = tokenize(question)
                question_uuid = qas[qid]['id']

                context_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in context_tokens]
                qustion_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in question_tokens]

                context_data.append(' '.join(context_ids))
                query_data.append(' '.join(qustion_ids))
                question_uuid_data.append(question_uuid)

    return context_data, query_data, question_uuid_data


# Same as in squad_preprocessing.py. Added uuid. Removed truths
def read_write_dataset_(dataset, tier, prefix):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""
    qn, an = 0, 0
    skipped = 0

    with open(os.path.join(prefix, tier +'.context'), 'w') as context_file,  \
         open(os.path.join(prefix, tier +'.question'), 'w') as question_file:#,\
         #open(os.path.join(prefix, tier +'.answer'), 'w') as text_file, \
         #open(os.path.join(prefix, tier +'.span'), 'w') as span_file:
        question_uuid_data = []
        for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
            article_paragraphs = dataset['data'][articles_id]['paragraphs']
            for pid in range(len(article_paragraphs)):
                context = article_paragraphs[pid]['context']
                # The following replacements are suggested in the paper
                # BidAF (Seo et al., 2016)
                context = context.replace("''", '" ')
                context = context.replace("``", '" ')

                context_tokens = tokenize(context)
                #answer_map = token_idx_map(context, context_tokens)

                qas = article_paragraphs[pid]['qas']
                for qid in range(len(qas)):
                    question = qas[qid]['question']
                    question_tokens = tokenize(question)
                    question_uuid = qas[qid]['id']
                    #answers = qas[qid]['answers']
                    qn += 1

                    num_answers = range(1)

                    for ans_id in num_answers:
                        # it contains answer_start, text
                        #text = qas[qid]['answers'][ans_id]['text']
                        #a_s = qas[qid]['answers'][ans_id]['answer_start']

                        #text_tokens = tokenize(text)

                        #answer_start = qas[qid]['answers'][ans_id]['answer_start']

                        #answer_end = answer_start + len(text)

                        #last_word_answer = len(text_tokens[-1]) # add one to get the first char

                        try:
                            #a_start_idx = answer_map[answer_start][1]

                            #a_end_idx = answer_map[answer_end - last_word_answer][1]

                            # remove length restraint since we deal with it later
                            context_file.write(' '.join(context_tokens) + '\n')
                            question_file.write(' '.join(question_tokens) + '\n')
                            #text_file.write(' '.join(text_tokens) + '\n')
                            #span_file.write(' '.join([str(a_start_idx), str(a_end_idx)]) + '\n')
                            question_uuid_data.append(question_uuid)
                        except Exception:# as e:
                            skipped += 1

                        an += 1

    print("Skipped {} question/answer pairs in {}".format(skipped, tier))
    return qn,an, question_uuid_data


# Same as in qa_data.py. Removed condition on file existence
def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None):
    #if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = qa_data.initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
        with gfile.GFile(target_path, mode="w") as tokens_file:
            counter = 0
            for line in data_file:
                counter += 1
                if counter % 5000 == 0:
                    print("tokenizing line %d" % counter)
                token_ids = qa_data.sentence_to_token_ids(line, vocab, tokenizer)
                tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_dev(prefix, dev_filename, vocab):
    # Don't check file size, since we could be using other datasets
    dev_dataset = maybe_download(squad_base_url, dev_filename, prefix)

    dev_data = data_from_json(os.path.join(prefix, dev_filename))
    context_data, question_data, question_uuid_data = read_dataset(dev_data, 'dev', vocab)

    return context_data, question_data, question_uuid_data


def generate_answers(sess, model, dataset, rev_vocab):
    """
    Loop over the dev or test dataset and generate answer.

    Note: output format must be answers[uuid] = "real answer"
    You must provide a string of words instead of just a list, or start and end index

    In main() function we are dumping onto a JSON file

    evaluate.py will take the output JSON along with the original JSON file
    and output a F1 and EM

    You must implement this function in order to submit to Leaderboard.

    :param sess: active TF session
    :param model: a built QASystem model
    :param rev_vocab: this is a list of vocabulary that maps index to actual words
    :return:
    """
    answers = {}
    answers_list = model._run_epoch_dev_part(sess, dataset['dev'])
    assert len(answers_list) == len(dataset['dev']['uuid'])
    for i in xrange(len(answers_list)):
        answers[dataset['dev']['uuid'][i]] = answers_list[i]
    return answers


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


#def adjust_dataset(dat, key0):
#    '''
#    adjusts the dataset dat to the old api
#    returns out, a nested dict
#    out[A][B]
#    A is key0
#    B can be '.ids.context'
#             '.ids.question'
#             '.ids.context.mask'
#             '.ids.question.mask'
#             '.ids.context.lens'
#             '.ids.question.lens'
#             'uuid'
#    '''
#    out = {}
#    out[key0] = {}
#    key1 = ['.ids.context', '.ids.question']
#    widths = [FLAGS.d_max_width, FLAGS.q_max_width]
#    for i in [0, 1]:
#        oldLens = []
#        manyLines = np.zeros((len(dat[i]), widths[i]), dtype=int)
#        for j, line in enumerate(dat[0]):
#            oldLen = line2nums(line, widths[i], manyLines[j, :])
#            oldLens.append(oldLen)    
#        out[key0][key1[i]] = manyLines
#        newLens = oldLensNewLens(oldLens, widths[i])
#        out[key0][key1[i]+'.lens'] = newLens
#        out[key0][key1[i]+'.mask'] = len2mask(oldLens, widths[i])
#    out[key0]['uuid'] = dat[2]
#    return out


def main(_):

    vocab, rev_vocab = initialize_vocab(FLAGS.vocab_path)

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    # ========= Load Dataset =========
    # You can change this code to load dataset in your own way

#    # Old version. Encoding issue
#    dev_dirname = os.path.dirname(os.path.abspath(FLAGS.dev_path))
#    dev_filename = os.path.basename(FLAGS.dev_path)
#    context_data, question_data, question_uuid_data = prepare_dev(dev_dirname, dev_filename, vocab)
#    dataset = (context_data, question_data, question_uuid_data)
#
#    dataset = adjust_dataset(dataset, 'dev')

    # New version. To match the preprocessing provided to train.py
    
    # As in squad_preprocessing.py:
    tt = time.time()
    download_prefix = os.path.join("download", "squad")
    data_prefix = os.path.join("data", "squad")
    if not os.path.exists(download_prefix):
        os.makedirs(download_prefix)
    if not os.path.exists(data_prefix):
        os.makedirs(data_prefix)
    dev_filename = "dev-v1.1.json"
    dev_data = data_from_json(os.path.join(download_prefix, dev_filename))
    (dev_num_questions,
     dev_num_answers,
     uuid) = read_write_dataset_(dev_data, 'dev', data_prefix)
    print("Processed {} questions and {} answers in dev".format(
            dev_num_questions, dev_num_answers))
    print(time.time()-tt, 'part: squar_preprocessing.py')

    # As in qa_data.py:
    tt = time.time()
    args = qa_data.setup_args()
    vocab_path = pjoin(args.vocab_dir, "vocab.dat")
    dev_path = pjoin(args.source_dir, "dev")
    x_dev_dis_path = dev_path + ".ids.context"
    y_dev_ids_path = dev_path + ".ids.question"
    data_to_token_ids(dev_path + ".context", x_dev_dis_path, vocab_path)
    data_to_token_ids(dev_path + ".question", y_dev_ids_path, vocab_path)
    print(time.time()-tt, 'part: qa_data.py')

    # As in train.py
    tt = time.time()
    dataset = {}
    load_data_dq(dataset, 'dev', FLAGS.data_dir)
    indices = trim_empty(dataset['dev'])
    print(time.time()-tt, 'part: train.py')
    
    # uuid from read_write_dataset_
    uuid = [i for j, i in enumerate(uuid) if j not in indices]
    dataset['dev']['uuid'] = uuid

    # sanity check
    keys = dataset['dev'].keys()
    for i in xrange(len(keys)-1):
        print(i, keys[i], len(dataset['dev'][keys[i]]),
              i+1, keys[i+1], len(dataset['dev'][keys[i+1]]))
        assert len(dataset['dev'][keys[i]]) == len(dataset['dev'][keys[i+1]])

    # ========= Model-specific =========
    # You must change the following code to adjust to your model

    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
    decoder = Decoder(output_size=FLAGS.output_size)

    with tf.Session() as sess:
        pass
    local_device_protos = device_lib.list_local_devices()  # 38559755
    for x in local_device_protos:
        if x.device_type == 'GPU':
            FLAGS.ifgpu = True
            break

    qa = QASystem(encoder, decoder, embed_path, rev_vocab)

    #with tf.Session() as sess:
    #    pass
    train_dir = get_normalized_train_dir(FLAGS.train_dir)
    initialize_model(sess, qa, train_dir)
    answers = generate_answers(sess, qa, dataset, rev_vocab)

    # write to json file to root dir
    with io.open('dev-prediction.json', 'w', encoding='utf-8') as f:
        f.write(unicode(json.dumps(answers, ensure_ascii=False)))


if __name__ == "__main__":
  tf.app.run()

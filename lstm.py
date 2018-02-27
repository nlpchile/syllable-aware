# coding: utf-8

## Imports
from src.RNN import RecurrentLSTM
from src.Corpus import Corpus
from src.utils import preprocessing_file
from src.perplexity import metric_pp
from src.Generators import GeneralGenerator

import time

import keras # para Callbacks

import losswise
from src.callback_losswise import LosswiseKerasCallback

import json

########################################################################################################################

## Setting Seed for Reproducibility
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development

import os
import numpy as np
import random

# import tensorflow as tf

os.environ['PYTHONHASHSEED'] = '57' # https://github.com/fchollet/keras/issues/850
seed = 57 # must be the same as PYTHONHASHSEED
np.random.seed(seed)
random.seed(seed)

## Limit operation to 1 thread for deterministic results.
# session_conf = tf.ConfigProto( intra_op_parallelism_threads = 1 , inter_op_parallelism_threads = 1 )
# from keras import backend as K
# tf.set_random_seed(seed)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)

########################################################################################################################

## Path to File

#path_in = './data/horoscopo_test_overfitting.txt'
#path_out = './data/horoscopo_test_overfitting_add_space.txt'

path_in = './data/train.txt'
path_out = './data/train_add_space.txt'


###################################################

## Pre processing
print('\n Preprocess - Add Spaces \n')

to_ignore = '''¡!()[]{}\"\'0123456789…-=@+*\t%&//­\xc2'''
signs_to_ignore = [i for i in to_ignore]

map_punctuation = {'¿': '<ai>',
                   '?': '<ci>',
                   '.': '<pt>',
                   '\n': '<nl>',
                   ',': '<cm>',
                   '<unk>': '<unk>',
                   ':': '<dc>',
                   ';': '<sc>'
                   }

letters = 'aáeéoóíúiuübcdfghjklmnñopqrstvwxyz'


add_space = True

if add_space:
    preprocessing_file(path_in=path_in,
                       path_out=path_out,
                       to_ignore=to_ignore
                       )

path_to_file = path_out


########################################################################################################################

## Hyperparameters

D = 512

recurrent_dropout = 0
dropout = 0

if keras.backend.backend() == 'tensorflow':
    recurrent_dropout = 0.3
    dropout = 0.3

dropout_seed = 1

train_size = 0.8 # 1
batch_size = 256
epochs = 300

optimizer = 'rmsprop' # 'adam'
metrics = ['top_k_categorical_accuracy', 'categorical_accuracy']

workers = 16 # default 1


################ CORPUS ATRIBUTES #################

T = 6000 # quantity of tokens

quantity_word = 2000
quantity_syllable = T - quantity_word

L = 100

random_split = True
token_split = '<nl>'
use_perplexity = True

###################################################

## Init Corpus
print('\n Starting Corpus \n')
corpus = Corpus(path_to_file=path_to_file,
                final_char=':',
                final_punc='>',
                inter_char='-',
                signs_to_ignore=signs_to_ignore,
                words_to_ignore=[],
                map_punctuation=map_punctuation,
                letters=letters,
                sign_not_syllable='<sns>'
                )
print('Start Corpus Done \n')


## Tokenization
print('\n Selecting Tokens \n')
corpus.set_tokens_selector(quantity_word=quantity_word,
                           quantity_syllable=quantity_syllable
                           )

token_selected = corpus.select_tokens_from_file(path_to_file = path_to_file)
print('Select Tokens Done\n')

print('\n Building Dictionaries \n')
corpus.build_dictionaries(token_selected = token_selected)
print('Build Dictionaries Done\n')


corpus.set_lprime(token_selected = token_selected, sequence_length = L)

params_corpus = corpus.params()

train_set, val_set = corpus.split_corpus(percentage = 80,
                                          random_split = random_split,
                                          token_split=token_split,
                                          min_len = 0
                                         )

print("average tokens per words = {}".format(params_corpus["average_tpw"]))
if use_perplexity: metrics.append(metric_pp(average_TPW = params_corpus["average_tpw"]))


######################## TEST COVERAGE ##################

cover = corpus.coverage(path_to_file)

print("With {} words and {} syllables the corpus coverage is {} percent".format(quantity_word,
                                                                                quantity_syllable,
                                                                                cover
                                                                                )
      )

########################################################################################################################

## Init Model
print('\n Init Model \n')
model = RecurrentLSTM(vocab_size=len(params_corpus["vocabulary"]),
                      embedding_dim=D,
                      hidden_dim=D,
                      input_length= params_corpus["lprime"],
                      recurrent_dropout=recurrent_dropout,
                      dropout=dropout,
                      seed=dropout_seed
                      )


## Build Model
print('\n Build Model \n')
model.build(optimizer=optimizer,
            metrics=metrics
            )


## Model Summary
print('\n Model Summary \n')
print(model.summary)


########################################################################################################################

## Generators
print('\n Get Generators \n')

train_generator = GeneralGenerator(batch_size = batch_size,
                                   ind_tokens = train_set,
                                   vocabulary = params_corpus["vocabulary"],
                                   max_len = params_corpus["lprime"],
                                   split_symbol_index = token_split,
                                   count_to_split = -1
                                   )

val_generator = GeneralGenerator(batch_size = batch_size,
                                 ind_tokens = train_set,
                                 vocabulary = params_corpus["vocabulary"],
                                 max_len = params_corpus["lprime"],
                                 split_symbol_index = token_split,
                                 count_to_split = -1
                                 )


######################### TEST SET ################################

path_to_test = './data/test.txt'

test_set = corpus.select_tokens_from_file(path_to_test)

test_generator = GeneralGenerator(batch_size = batch_size,
                                 ind_tokens = train_set,
                                 vocabulary = params_corpus["vocabulary"],
                                 max_len = params_corpus["lprime"],
                                 split_symbol_index = token_split,
                                 count_to_split = -1
                                 )

########################################################################################################################

## Callbacks
# https://keras.io/callbacks/

out_directory_train_history = './train_history/'
out_directory_model = './models/'
out_model_pref = 'lstm_model_'


if not os.path.exists(path=out_directory_model):
    os.mkdir(path=out_directory_model,
             mode=0o755
             )
else:
    pass

if not os.path.exists(path=out_directory_train_history):
    os.mkdir(path=out_directory_train_history,
             mode=0o755
             )
else:
    pass


time_pref = time.strftime('%y%m%d.%H%M') # Ver código de Jorge Perez

outfile = out_model_pref + time_pref + '.h5'


###################################################

# Checkpoint
# https://keras.io/callbacks/#modelcheckpoint

monitor_checkpoint = 'val_top_k_categorical_accuracy' # 'val_loss'


checkpoint = keras.callbacks.ModelCheckpoint(filepath=out_directory_model + outfile,
                                             monitor=monitor_checkpoint,
                                             verbose=1,
                                             save_best_only=True, # TODO: Guardar cada K epochs, y Guardar el mejor
                                             save_weights_only=False,
                                             mode='auto',
                                             period=1 # Interval (number of epochs) between checkpoints.
                                             )


###################################################

## EarlyStopping
# https://keras.io/callbacks/#earlystopping

monitor_early_stopping = 'val_top_k_categorical_accuracy' # 'val_loss'

patience = 100 # number of epochs with no improvement after which training will be stopped


early_stopping = keras.callbacks.EarlyStopping(monitor=monitor_early_stopping,
                                               min_delta=0,
                                               patience=patience,
                                               verbose=0,
                                               mode='auto'
                                               )


###################################################

## Losswise


keyfile = json.load(open('.env'))

losswise_api_key = keyfile["losswise_api_key"]
losswise_tag = keyfile["losswise_tag"] + " T = {} ; Tw = {} ; Ts = {}"

losswise_tag = losswise_tag.format(T, quantity_word, quantity_syllable)

losswise.set_api_key(losswise_api_key)

params_data = json.loads(model.to_json)

params_data['samples'] = len(train_generator.ind_tokens)
params_data['steps'] = train_generator.steps_per_epoch

params_data['batch_size'] = train_generator.batch_size # para meterlo igual a los params que se muestran online

params_model = {'batch_size': train_generator.batch_size}

losswise_callback = LosswiseKerasCallback(tag=losswise_tag,
                                          params_data=params_data,
                                          params_model=params_model
                                          )

losswise_callback.set_params(params=params_model)


###################################################

## Callbacks Pipeline
callbacks = [checkpoint, early_stopping, losswise_callback]


########################################################################################################################

## Training
print('\n Training \n')
ti = time.time()


model.fit(train_generator=train_generator,
          val_generator=val_generator,
          epochs=epochs,
          callbacks=callbacks,
          workers=workers
          )


tf = time.time()
dt = (tf - ti) / 60.0
print('\n Elapsed Time {} \n'.format(dt))


# iter = 0
# time_pref = time_pref[:-1] + str(i)

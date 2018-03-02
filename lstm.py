# coding: utf-8

# PYTORCH

## Imports
import time
import os
import numpy as np
import random

from src.RNN import RecurrentLSTM
from src.Tokenization import Tokenization
from src.utils import preprocessing_file
from src.Generators import GeneralGenerator

########################################################################################################################
## Setting Seed for Reproducibility

os.environ['PYTHONHASHSEED'] = '57' # https://github.com/fchollet/keras/issues/850
seed = 57 # must be the same as PYTHONHASHSEED
np.random.seed(seed)
random.seed(seed)

########################################################################################################################

## Path to File

#path_in = './data/horoscopo_test_overfitting.txt'
#path_out = './data/horoscopo_test_overfitting_add_space.txt'

#path_in = './data/nicanor_clear.txt'
#path_out = './data/nicanor_clear2.txt'

path_in = './data/train.txt'
path_out = './data/train_add_space.txt'


##################################################

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

recurrent_dropout = 0.3
dropout = 0.3

train_size = 0.8 # 1

batch_size = 128
epochs = 300


################ CORPUS ATRIBUTES #################

T = 500 # quantity of tokens

quantity_word = 30
quantity_syllable = T - quantity_word

L = 100  # 100 sequence_length

random_split = False
token_split = '<nl>'


###################################################

## Init Corpus
print('\nStarting Corpus \n')
tokenization = Tokenization(path_to_file=path_to_file,
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
print('\nSelecting Tokens \n')
tokenization.setting_tokenSelector_params(quantity_word=quantity_word,
                                          quantity_syllable=quantity_syllable
                                          )

token_selected = tokenization.select_tokens()
print('Select Tokens Done\n')

print('\nSetting experiment\n')
tokenization.setting_experiment(token_selected=token_selected,
                                sequence_length=L
                                )

print('Set experiment Done\n')

print("\nGet and save parameters experiment")
params_tokenization = tokenization.params_experiment()

path_setting_experiment = "./data/experimentT{}Tw{}Ts{}.txt".format(T,
                                                                    quantity_word,
                                                                    quantity_syllable
                                                                    )

tokenization.save_experiment(path_setting_experiment)

train_set, val_set = tokenization.split_train_val(train_size=train_size,
                                                  random_split=random_split,
                                                  token_split=token_split,
                                                  min_len=0
                                                  )

print("size train set = {}, size val set = {}".format(len(train_set), len(val_set)))


######################## TEST COVERAGE ##################

words_cover_with_words, words_cover_with_syll, sylls_cover_with_syll = tokenization.coverage(path_to_file)
text = "With {} words the words corpus coverage is {} percent \nWith {} syllables the words corpus coverage is {} and the syllables cover is {}"
print(text.format(quantity_word,
                  words_cover_with_words,
                  quantity_syllable,
                  words_cover_with_syll,
                  sylls_cover_with_syll
                  )
      )


########################################################################################################################

## Init Model
print('\n Init Model \n')
model = RecurrentLSTM(vocab_size=len(params_tokenization["vocabulary"]),
                      embedding_dim=D,
                      hidden_dim=D,
                      input_length=params_tokenization["lprime"],
                      recurrent_dropout=recurrent_dropout,
                      dropout=dropout,
                      )


## Build Model


########################################################################################################################

## Generators
print('\n Get Generators \n')


if params_tokenization["lprime"] > len(train_set):
    raise ValueError("lprime > len(train_set), lprime = {} and len(train_set) = {}".format(params_tokenization["lprime"], len(train_set)))

train_generator = GeneralGenerator(batch_size=batch_size,
                                   ind_tokens=train_set,
                                   vocabulary=params_tokenization["vocabulary"],
                                   max_len=params_tokenization["lprime"],
                                   split_symbol_index=token_split,
                                   count_to_split=-1,
                                   ).__next__()


if params_tokenization["lprime"] > len(val_set):
    raise ValueError("lprime > len(val_set), lprime = {} and len(val_set) = {}".format(params_tokenization["lprime"], len(val_set)))
                     
val_generator = GeneralGenerator(batch_size=batch_size,
                                 ind_tokens=val_set,
                                 vocabulary=params_tokenization["vocabulary"],
                                 max_len=params_tokenization["lprime"],
                                 split_symbol_index=token_split,
                                 count_to_split=-1
                                 ).__next__()



########################################################################################################################

samples = len(train_set)
steps_per_epoch = samples / batch_size
batch_size = batch_size


########################################################################################################################

## Training
print('\n Training \n')
ti = time.time()

# insert fit

tf = time.time()
dt = (tf - ti) / 60.0
print('\n Elapsed Time {} \n'.format(dt))

from TokenSelector import TokenSelector
from utils import Lprime


class Corpus:
  
    def __init__(self,
               path_to_file,
               train_size,
               final_char=':',
               final_punc='>',
               inter_char='-',
               sign_to_ignore=[],
               word_to_ignore=[]):
        
        self.path_to_file = path_to_file
        self.train_size = train_size
        self.final_char = final_char
        self.final_punc = final_punc
        self.inter_char = inter_char
        self.sign_to_ignore = sign_to_ignore
        self.word_to_ignore = word_to_ignore

        self.tokenSelector = TokenSelector(final_char = self.final_char,
                           inter_char = self.inter_char,
                           sign_to_ignore = self.sign_to_ignore,
                           word_to_ignore= self.word_to_ignore)

        self.tokenSelector.get_dictionary(self.path_to_file)


    def select_tokens(self, quantity_word, quantity_syllable):
        
        self.quantity_word = quantity_word
        self.quantity_syllable = quantity_syllable
        self.tokenSelector.get_frequent(quantity_word = self.quantity_word,
                                        quantity_syll = self.quantity_syllable)

        self.token_selected = []
        with open(self.path_to_file) as f1:
                for line in f1:
                    words = line.lower().split()
                    for token in words:
                        token = token.strip()
                        tokenSelector.select(token, self.token_selected)


    def calculateLprime(self, sequent_length):
        self.lprime = Lprime(token_selected, sequent_length)
    
    
    def dictionaries_token_index(self):
        
        self.vocabulary = set(self.token_selected)
        self.token_to_index = dict((t, i) for i, t in enumerate(self.vocabulary, 1))
        self.indice_ends = ending_tokens_index(self.token_to_index, [self.final_char, self.final_punc])
        self.index_to_token = dict((self.token_to_index[t], t) for t in self.vocabulary)
        self.ind_corpus = [self.token_to_index[token] for token in self.tokens] # corpus as indexes
        self.vocabulary_as_index = set(self.ind_corpus) # vocabualry as index
 
        len_train = int(len(self.ind_corpus)*self.train_size)
        self.train_set = self.ind_corpus[0:len_train] # indexes
        self.test_set = self.ind_corpus[len_train:] # indexes
        self.vocabulary_train = set(self.train_set) # indexes
        self.vocabulary_test = set(self.test_set) # indexes


    def split_train_eval(val_percentage, token_split, min_len = 0):

        self.vocabulary = set(self.token_selected)
        self.token_to_index = dict((t, i) for i, t in enumerate(self.vocabulary, 1))

        self.indice_ends = ending_tokens_index(self.token_to_index, [self.final_char, self.final_punc])

        self.index_to_token = dict((self.token_to_index[t], t) for t in self.vocabulary)
        self.ind_corpus = [self.token_to_index[token] for token in self.tokens] # corpus as indexes
        self.vocabulary_as_index = set(self.ind_corpus) # vocabualry as index

        self.train_set = []
        self.eval_set = []
        tokens = []

        self.token_selected = self.token_selected if self.token_selected[-1] == token_split else aux + [token_split]

        self.tokensplit = self.token_to_index[token_split]

        words_train_set = 0
        words_eval_set = 0
        qw = 0

        for token in self.ind_corpus:
            if token in self.indice_ends:
                qw += 1
            if token == self.tokensplit:
                if len(tokens) < min_len:
                    tokens = []
                    continue
                p = random.choice(range(0, 100))
                if p < val_percentage:
                    self.eval_set += tokens + [self.tokensplit]
                    words_eval_set += qw + 1
                    qw = 0
                else:
                    self.train_set += tokens + [self.tokensplit]
                    words_train_set += qw + 1
                    qw = 0
                tokens = []
            else:
                tokens.append(token)

        self.train_ATPW = words_train_set / len(train_set)
        self.eval_ATPW = words_eval_set / len(eval_set)
from .utils import tokenize_corpus, get_most_frequent


class TokenSelector():

    def __init__(self,
                 final_char,
                 inter_char,
                 signs_to_ignore,
                 words_to_ignore,
                 map_punctuation,
                 letters
                 ):

        self.final_char = final_char
        self.inter_char = inter_char
        self.signs_to_ignore = signs_to_ignore
        self.words_to_ignore = words_to_ignore

        # Punctuation
        self.map_punctuation = map_punctuation
        self.punctuation = set(self.map_punctuation)

        # Characters
        self.characters = set(letters)


    def get_dictionary(self, path_file):

        to_ignore = [i for i in self.punctuation]
        to_ignore = to_ignore + self.signs_to_ignore + self.words_to_ignore

        self.dict_word, self.dict_syll, self.freq_word, self.freq_syll = tokenize_corpus(path_file = path_file,
                                                                                         to_ignore = to_ignore
                                                                                         )


    def get_frequent(self, quantity_word, quantity_syll):

        self.words = get_most_frequent(freq_dict = self.freq_word,
                                       quantity = quantity_word,
                                       to_ignore = []
                                       )

        # count syllables ignoring most frequent words

        syll_to_ignore = []

        for w in self.words:
            syll_to_ignore += self.dict_word[w]

        self.syllables = get_most_frequent(freq_dict =  self.freq_syll,
                                           quantity = quantity_syll,
                                           to_ignore = syll_to_ignore
                                           )


    def select(self, token, tokens_selected):

        if token in self.punctuation:
            tokens_selected.append(self.map_punctuation[token])

        elif token in self.dict_word:

            if token in self.words:
                tokens_selected.append(token + self.final_char)

            else:
                for s in self.dict_word[token]:

                    if s in self.syllables:
                        tokens_selected.append(s)

                    else:
                        for c in self.dict_syll[s]:
                            if c in self.characters:
                                tokens_selected.append(c)

        return tokens_selected

from src.CheckModel import CheckModel


def main():

    '''
    print("="*50)
    print("Loading files ...")

    path_model = "../models/lstm_model_train2_experimentT10000Tw100Ts2850180312.0830_2.17_2.12_30.h5"
    path_experiment = "../models/experiment/train2_experimentT10000Tw100Ts2850_setting_tokenize.txt"
    path_tokenselector = "../models/experiment/train2_experimentT10000Tw100Ts2850_setting_tokenSelector.txt"
    path_test = "./data/test2.txt"

    checkmodel = CheckModel(path_model, path_experiment, path_tokenselector)

    print("="*50)
    print("computing perplexity per word using file '{}'".format(path_test))
    ppl = checkmodel.perplexity(path_test)
    print("perplexity per word was '{}'".format(ppl))
    '''

def separate_files():


    # ls ~/ribanez/models >> ~/models_IDs.txt
    # cat ~/models_IDs.txt

    # ls ~/ribanez/models/experiment >> ~/experiments_IDs.txt
    # cat ~/experiments_IDs.txt

    file = open('/home/nlp/experiments_IDs.txt', 'r')
    lines = file.readlines()

    ##

    file_1 = open('/home/nlp/file_tokenize.txt', 'w')
    file_2 = open('/home/nlp/file_tokenSelector.txt', 'w')

    i = 0

    for line in lines:

        if i == 0:

            file_1.write(line)

            i += 1

        else:

            file_2.write(line)

            i -= 1

    ##


if __name__ == '__main__':

    # main()
    separate_files()

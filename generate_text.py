from src.CheckModel import CheckModel


def main():

    print("="*50)
    print("Loading files ...")
    path_model = "../models/"
    path_experiment = "../models/experiment/train2_experimentT500Tw30Ts470_setting_tokenize.txt"
    path_tokenselector = "../models/experiment/train2_experimentT500Tw30Ts470_setting_tokenSelector.txt"
    path_test = "./data/test2.txt"

    checkmodel = CheckModel(path_model, path_experiment, path_tokenselector)
    seed = "declaro"

    print("="*50)
    print("predicting text using '{}' like seed".format(seed))
    text = checkmodel.predict_text(nwords=20)

    print("="*50)
    print("text generate '{}'".format(text))
    print("="*50)
    print("computing perplexity per word using file '{}'".format(path_test))
    ppl = checkmodel.perplexity(path_test)
    print("perplexity per word was '{}'".format(ppl))

if __name__ == '__main__':
    main()
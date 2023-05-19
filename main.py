from neural import Mod, Classifier

if __name__ == '__main__':
    # Budowa modelu
    mod = Mod('./datasets/train_set')
    # Klasyfikacja
    classifier = Classifier('./datasets/test_set', 'v02_rnn.tf', 'v01_cnn.h5', filename='results/results.txt')
    classifier.classify_data()

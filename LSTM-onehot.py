# from __future__ import print_function

import jieba
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding, Dense, Activation, LSTM
from keras.optimizers import RMSprop
import numpy as np
import sys
import os
import glob

folder_path = ".\\data"  # 文件夹路径
file_extension = "*.txt"  # 文件扩展名


class LanguageModel:
    def __init__(self, step=3, embed_size=128, seq_length=20):

        self.seq_length = seq_length
        self.step = step
        self.embed_size = embed_size

    def load_data(self, path):
        english = u'[a-zA-Z0-9’!"#$%&\'()*+-./:：「<=>?@★…【】《》“”‘’！[\\]^_`{|}~+Ｓａｅｋｍｎｏ]'
        str1 = u'[○①②③④⑤⑥⑦⑧⑨⑩_“”《》‘’」「…『』（）<>【】．·.—*-~﹏]'
        remove_set = set(english + str1)

        # read the entire text
        file_paths = glob.glob(os.path.join(folder_path, file_extension))
        text = ""
        for path in file_paths:
            with open(path, "r", encoding='ANSI') as file:
                temp = file.read()
                result = filter(lambda x: x not in remove_set, temp)
                new_temp = ''.join(result)
                new_temp = new_temp.replace("\n", '')
                new_temp = new_temp.replace("\u3000", '')
                new_temp = new_temp.replace(" ", '')
                text = text + new_temp

        print('corpus length:', len(text))

        # all the vocabularies
        text = list(jieba.cut(text))
        vocab = sorted(list(set(text)))
        print('total words:', len(vocab))
        # create word-index dict
        word_to_index = dict((c, i) for i, c in enumerate(vocab))
        index_to_word = dict((i, c) for i, c in enumerate(vocab))
        # cut the text into fixed size sequences
        sentences = []
        next_words = []
        for i in range(0, len(text) - self.seq_length, self.step):
            sentences.append(list(text[i:i + self.seq_length]))
            next_words.append(text[i + self.seq_length])
        print('nb sequences:', len(sentences))

        # generate training samples
        X = np.asarray([[word_to_index[w] for w in sent[:]] for sent in sentences])
        y = np.zeros((len(sentences), len(vocab)))
        for i, word in enumerate(next_words):
            y[i, word_to_index[word]] = 1

        self.text = text
        self.vocab = vocab
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
        self.X = X
        self.y = y

    def load_model(self):
        # load a Sequential model
        model = Sequential()
        model.add(Embedding(len(self.vocab), self.embed_size, input_length=self.seq_length))
        model.add(LSTM(self.embed_size, input_shape=(self.seq_length, self.embed_size), return_sequences=False))
        model.add(Dense(len(self.vocab)))
        model.add(Activation('softmax'))
        self.model = model
        # compile the model
        optimizer = RMSprop(lr=0.005)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def fit_model(self, batch_size=128, nb_epoch=5):
        # fit the model with trainind data
        callbacks_list = [
            keras.callbacks.ModelCheckpoint(  # 在每轮完成后保存权重
                filepath='lstm_model_onehot.h5',
                monitor='loss',
                save_best_only=True,
            ),
            keras.callbacks.ReduceLROnPlateau(  # 不再改善时降低学习率
                monitor='loss',
                factor=0.5,
                patience=1,
            ),
            keras.callbacks.EarlyStopping(  # 不再改善时中断训练
                monitor='loss',
                patience=3,
            ),
        ]
        self.model.fit(self.X, self.y, batch_size=128, epochs=nb_epoch, callbacks=callbacks_list)

    def save(self, path):
        self.model.save(path)

    def predict(self, x, verbose=0):
        return self.model.predict([x], verbose=verbose)[0]

    def _sample(self, preds, diversity=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / diversity
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def generate_text(self):
        # generate text from random text seed
        # start_index = random.randint(0, len(self.text) - self.seq_length - 1)

        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('--------diversity:', diversity)

            generated = ''
            sentence = self.text[0:0 + self.seq_length]
            # print(sentence)
            # sentence = '江苏与浙江到宋朝时已渐渐成为中国的经济与文化中心，苏州、杭州成为出产文化和美女的地方。'
            # sentence = list(jieba.cut(sentence))
            for word in sentence:
                generated += word
            print('--------Generating with seed:', generated)
            sys.stdout.write(generated)
            for i in range(50):
                x = np.asarray([self.word_to_index[w] for w in sentence]).reshape([1, self.seq_length])
                preds = self.predict(x)
                next_index = self._sample(preds, diversity)
                next_word = self.index_to_word[next_index]

                generated += next_word
                sentence.append(next_word)
                sentence = sentence[1:]

                sys.stdout.write(next_word)
                sys.stdout.flush()
            print()


if __name__ == '__main__':
    model = LanguageModel(seq_length=50, embed_size=256)
    model.load_data('')
    model.load_model()
    model.fit_model(nb_epoch=20)
    model.save("./LSTM_onehot_model.h5")

    for i in range(1, 3):
        print('Iteration:', i)
        model.generate_text()
        print()
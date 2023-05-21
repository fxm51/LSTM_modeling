import os
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from gensim.models.word2vec import Word2Vec
import os
import sys
import glob
import jieba

folder_path = ".\\data"  # 文件夹路径
file_extension = "*.txt"  # 文件扩展名


def preprocess_data():
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
    _corpus = [list(jieba.cut(text))]
    return _corpus


class Model:
    def __init__(self, step=3, embed_size=128, seq_length=20):
        self.LSTM_model = None
        self.w2v_model = None
        self.seq_length = seq_length
        self.step = step
        self.embed_size = embed_size

    def build_w2v_model(self, corpus):
        w2v_model = Word2Vec(corpus, vector_size=128, window=5, min_count=1, workers=4)
        self.w2v_model = w2v_model
        w2v_model.wv.save_word2vec_format('word2vec.txt')

    def build_LSTM_model(self):
        raw_input = [item for sublist in corpus for item in sublist]

        text_stream = []
        vocab = self.w2v_model.wv.key_to_index
        for word in raw_input:
            if word in vocab:
                text_stream.append(word)

        seq_length = 20
        x = []
        y = []
        for i in range(0, len(text_stream) - seq_length):
            given = text_stream[i:i + seq_length]
            predict = text_stream[i + seq_length]
            x.append(np.array([self.w2v_model.wv[word] for word in given]))
            y.append(self.w2v_model.wv[predict])

        x = np.array(x)
        y = np.array(y)
        x = np.reshape(x, (-1, seq_length, 128))
        y = np.reshape(y, (-1, 128))

        lstm_model = Sequential()
        lstm_model.add(LSTM(256, input_shape=(seq_length, 128), return_sequences=False))
        lstm_model.add(Dense(128, activation='sigmoid'))
        lstm_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        callbacks_list = [
            keras.callbacks.ModelCheckpoint(  # 在每轮完成后保存权重
                filepath='LSTM_word2vec_model.h5',
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
        # 跑模型
        lstm_model.fit(x, y, epochs=20, batch_size=4096, verbose=1, callbacks=callbacks_list)
        self.LSTM_model = lstm_model
        # lstm_model.save("./LSTM_word2vec_model.h5")

    def predict_next(self, input_array):
        input_array = np.array(input_array)
        x = np.reshape(input_array, (-1, self.seq_length, 128))
        y = self.LSTM_model.predict(x, verbose=0)
        return y

    def _sample(self, preds, diversity=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / diversity
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def generate_article(self, _input, rounds=50):
        # input string to array
        text = [list(jieba.cut(_input))]
        len(text)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('--------diversity:', diversity)
            print('--------Generating with seed:', _input)
            print(_input, end='')
            for i in range(rounds):
                res = []
                for word in text[(len(text) - self.seq_length):]:
                    res.append(self.w2v_model.wv[word])
                predict_y = self.predict_next(res)[0]
                next_index = self._sample(predict_y)
                next_word = self.w2v_model.wv.index_to_key[next_index]
                next_word = "".join(next_word)
                print(next_word, end='')
                sys.stdout.flush()
        return _input


if __name__ == '__main__':
    corpus = preprocess_data()
    model = Model()
    model.build_w2v_model(corpus)
    model.build_LSTM_model()
    sentence = '那汉子左边背心上却插著一枝长箭。鲜血从他背心流到马背上，又'
    model.generate_article(_input=sentence)


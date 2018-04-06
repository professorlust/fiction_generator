'''使用LSTM语言模型生成文本，字符级的
'''

from __future__ import print_function
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.models import load_model
import argparse
import numpy as np
import random
import sys
import os

def read_dataset(maxlen=40, step=3):
    # 读取作品的文本数据
    path = 'datasets'
    files = os.listdir(path)
    text = ''
    for file in files:
        if not os.path.isdir(file):
            text += open(path+'/'+file, 'r').read().strip()

    # 生成字符词汇表，字符与索引之间的映射
    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    # 将文本切割为半冗余的序列，长度为 maxlen 个字符
    sentences = []  # 句子
    next_chars = []  # 句子的下一个字符
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sentences:', len(sentences))

    # 向量化操作
    # X: [nb_sequences, maxlen， len(chars)]，字符的 one-hot 表示
    # y：[nb_sequences, len(chars)]，字符的 one-hot 表示
    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    return text, len(chars), char_indices, indices_char, X, y

class LSTMgen():
    def __init__(self):
        self.maxlen = 40
        self.step = 3
        text, vocab_size, char_indices, indices_char, X, y = read_dataset(maxlen=self.maxlen, step=self.step)
        self.text = text
        self.vocab_size = vocab_size
        self.char_indices = char_indices
        self.indices_char = indices_char
        self.X = X
        self.y = y
        self.iteration = 1000

        if os.path.exists('saved_model.h5'):
            self.model = load_model('saved_model.h5')
        else:
            # 由于在这个脚本中，是以字符级别的 LSTM 来生成文本，
            # 字符词汇表非常小，所以并没有使用词嵌入
            # 但是处理其他的文本，例如中文，或者单词级别的 LSTM 的话，
            # 词汇表相当大，这个时候使用词嵌入进行优化是非常合适的。
            # build the model: a single LSTM
            # 构建模型，使用单层 LSTM 循环神经网络
            print('Build model...')
            model = Sequential()
            # 输入为一个序列，输出为 128维的向量，即 LSTM 的最终状态
            model.add(LSTM(128, input_shape=(self.maxlen, len(self.char_indices))))
            model.add(Dense(self.vocab_size))  # 全连接层，长度为词汇表大小
            model.add(Activation('softmax'))  # softmax激活层，检查下一个最有可能的字符
            model.summary()
            optimizer = RMSprop(lr=0.01)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer)
            self.model = model

    # 根据预测输入抽样
    @staticmethod
    def sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        # 从预测结果进一步抽样下一个字符，这里 temperature 决定了输出的多样性
        # 具体可以参照以下两篇中的解释
        # http://home.deib.polimi.it/restelli/MyWebSite/pdf/rl5.pdf
        # http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node17.html
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def train(self):
        # train the model, output generated text after each iteration
        # 训练模型，在每一次迭代后输出生成的文本
        for i in range(1, self.iteration):
            print()
            print('-' * 50)
            print('Iteration', i)
            self.model.fit(self.X, self.y, batch_size=128, nb_epoch=1)  # 训练
            if i%5==0:
                self.model.save('saved_model.h5')
            self.inference()


    def inference(self, seed=None):
        if seed is None:
            # 随机选择起始位置，以这个起始位置的文本为基础生成后续文本
            start_index = random.randint(0, len(self.text) - self.maxlen - 1)
            # 根据选择的起始位置检出起始文本，这里称为种子文本
            seed = self.text[start_index: start_index + self.maxlen]

        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)
            print('----- Generating with seed: "' + seed + '"')
            generated = seed
            sentence = seed
            sys.stdout.write(seed)  # 打印文本

            for i in range(400):  # 连续生成 400 个后续字符
                x = np.zeros((1, self.maxlen, self.vocab_size))
                for t, char in enumerate(sentence):
                    x[0, t, self.char_indices[char]] = 1.

                preds = self.model.predict(x, verbose=0)[0]  # 预测下一个结果
                next_index = self.sample(preds, diversity)  # 抽样出下一个字符的索引值
                next_char = self.indices_char[next_index]  # 检出下一个字符

                generated += next_char
                sentence = sentence[1:] + next_char  # 输入后移一格

                sys.stdout.write(next_char)  # 连续打印
                sys.stdout.flush()  # 刷新控制台
            print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=str, default=None, help='generating with seed')
    args = parser.parse_args()
    lstm_gen = LSTMgen()
    if args.seed:
        lstm_gen.inference(args.seed)
    else:
        lstm_gen.train()
# coding=utf-8
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pickle
from loadconfig.LoadConfig import LoadConfig
from preproc.SplitWord import read_train_data, load_stopwords
from model.ComPredModel import ComPredModel
import datetime
import matplotlib.pyplot as plt
import math
import sys


class Train:
    def __init__(self):
        # 加载配置文件，获取训练参数
        self.config = LoadConfig.config
        # 读取训练数据并分词
        stopwords = load_stopwords(self.config["stop_word"])
        samples, self.labels = read_train_data(self.config["data_path"], self.config["samples_path"],
                                               self.config["labels_path"], stopwords, self.config["text_clean"])
        # 划分训练集和验证集
        train_size = int(len(samples) * self.config["train_portion"])
        valid_size = int((len(samples) - train_size)/2)
        self.train_samples = samples[0: train_size]
        self.train_labels = self.labels[0: train_size]
        self.validation_samples = samples[train_size:train_size+valid_size]
        self.validation_labels = self.labels[train_size:train_size+valid_size]
        self.test_samples = samples[train_size+valid_size:]
        self.test_labels = self.labels[train_size+valid_size:]

    def word2vect(self):
        # 生成训练词向量
        if os.path.exists(self.config["tokenizer"]):
            with open(self.config["tokenizer"], 'rb') as handle:
                tokenizer = pickle.load(handle)
        else:
            tokenizer = Tokenizer(num_words=self.config["vocab_size"], oov_token=self.config["oov_tok"])
            tokenizer.fit_on_texts(self.train_samples)
            with open(self.config["tokenizer"], 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        train_sequences = tokenizer.texts_to_sequences(self.train_samples)
        train_padded = pad_sequences(train_sequences, maxlen=self.config["max_length"], padding=self.config["padding_type"],
                                     truncating=self.config["trunc_type"])
        validation_sequences = tokenizer.texts_to_sequences(self.validation_samples)
        validation_padded = pad_sequences(validation_sequences, maxlen=self.config["max_length"],
                                          padding=self.config["padding_type"], truncating=self.config["trunc_type"])
        test_sequences = tokenizer.texts_to_sequences(self.test_samples)
        test_padded = pad_sequences(test_sequences, maxlen=self.config["max_length"],
            padding=self.config["padding_type"], truncating=self.config["trunc_type"])
        training_label_seq = np.array(self.train_labels).reshape(len(self.train_labels), 1)
        validation_label_seq = np.array(self.validation_labels).reshape(len(self.validation_labels), 1)
        test_label_seq = np.array(self.test_labels).reshape(len(self.test_labels), 1)
        return train_padded, training_label_seq, validation_padded, validation_label_seq, test_padded, test_label_seq

    def train(self):
        train_padded, training_label_seq, validation_padded, validation_label_seq, _, _ = self.word2vect()
        if self.config["select_model"] == "cnn_regression":
            model = ComPredModel.cnn_regression(self.config)
        earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
        save_path = os.path.join(self.config["save_model"])
        mcp_save = ModelCheckpoint(filepath=save_path, save_best_only=True, monitor='val_loss', mode='min')
        log_dir = os.path.join('output', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))  # 在tensorboard可视化界面中会生成带时间标志的文件
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
        history = model.fit([train_padded], training_label_seq, validation_data=([validation_padded], validation_label_seq),
            epochs=self.config["num_epochs"], callbacks=[earlyStopping, mcp_save, tensorboard_callback], batch_size=self.config["batch_size"])
        model.save(self.config["save_model"])

    def penalty_mat_plot(self, gap):
        penalty_dict = {}
        for penalty in self.labels:
            counter = int(penalty/gap)
            if counter in penalty_dict:
                penalty_dict[counter] += 1
            else:
                penalty_dict[counter] = 1
        penalty_list = []
        counter_list = []
        for penalty, counter in penalty_dict.items():
            penalty_list.append(penalty * gap)
            counter_list.append(counter)
        plt.bar(penalty_list, counter_list, label='刑期分布')
        plt.legend()
        plt.xlabel('number')
        plt.ylabel('value')
        plt.show()

    def error_mat_plot(self, test_label, error, threshold, gap):
        test_label = test_label.reshape(test_label.size).tolist()
        error = error.reshape(error.size).tolist()
        error_dict = {}
        for count in range(len(test_label)):
            counter = int(test_label[count] / gap)
            if counter in error_dict and abs(error[count]) > threshold:
                error_dict[counter] += 1
            elif counter not in error_dict and abs(error[count]) > threshold:
                error_dict[counter] = 1
        penalty_list = []
        counter_list = []
        for penalty, counter in error_dict.items():
            penalty_list.append(penalty * gap)
            counter_list.append(counter)
        plt.bar(penalty_list, counter_list, label='error distribution')
        plt.legend()
        plt.xlabel('number')
        plt.ylabel('value')
        plt.show()

    def predict(self):
        with open(self.config["tokenizer"], 'rb') as handle:
            tokenizer = pickle.load(handle)
        model = None
        if self.config["select_model"] == "cnn_regression":
            model = ComPredModel.cnn_regression(self.config)
        model.load_weights(self.config["save_model"])
        _, _, _, _, test_padded, _= self.word2vect()
        pred = model.predict([test_padded])

    def evaluate(self):
        with open(self.config["tokenizer"], 'rb') as handle:
            tokenizer = pickle.load(handle)
        model = None
        if self.config["select_model"] == "cnn_regression":
            model = ComPredModel.cnn_regression(self.config)
        model.load_weights(self.config["save_model"])
        _, _, _, _, test_padded, test_label_seq = self.word2vect()
        pred = model.predict([test_padded])
        error = abs(pred - test_label_seq)/test_label_seq
        print(np.sum(abs(error)<=0.25)/error.size)


if __name__=="__main__":
    train = Train()
    if sys.argv[1] == "train":
        train.train()
    elif sys.argv[1] == "predict":
        train.predict()
    elif sys.argv[1] == "evaluate":
        train.evaluate()

import re
import jieba
import json
import random
import os
import numpy as np


def load_stopwords(filename):
    stopwords = []
    with open(filename, "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.replace("\n", "")
            stopwords.append(line)
    return stopwords


def str_find_list(string, words):
    for word in words:
        if string.find(word) != -1:
            return True
    return False


def text_cleaner(text, stop_words):
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning

    ]

    # text = text.replace(".", "")
    text = text.replace("[", " ")
    text = text.replace(",", " ")
    text = text.replace("]", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("\"", "")
    text = text.replace("-", " ")
    text = text.replace("=", " ")
    text = text.replace("?", " ")
    text = text.replace("!", " ")

    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
        text = text.strip()
    text = text.replace('+', ' ').replace(',', ' ').replace(':', ' ')
    text = re.sub("([0-9]+[年月日])+", "", text)
    text = re.sub("[a-zA-Z]+", "", text)
    text = re.sub("[0-9\.]+元", "", text)
    stop_words_user = ["年", "月", "日", "时", "分", "许", "某", "甲", "乙", "丙"]
    word_tokens = jieba.cut(text)
    text = [w for w in word_tokens if w not in stop_words if not str_find_list(w, stop_words_user)
            if len(w) >= 1 if not w.isspace()]
    return " ".join(text)


def read_train_data(filename, samples_path, labels_path, stopwords, text_clean=False):
    samples = []
    labels = []
    if not os.path.exists(samples_path) or not os.path.exists(labels_path):
        with open(filename, "r", encoding="utf-8") as fr:
            count = 0
            for line in fr:
                try:
                    line_json = json.loads(line)
                    label = line_json["judge"]
                    justice = line_json["justice"]
                    justice += line_json["opinion"]
                except:
                    justice = line.split("justice")[1].split("opinion")[0]
                    justice = justice.replace(':', '').replace('"', '')
                    opinion = line.split("opinion")[1].split("province")[0]
                    opinion = opinion.replace(':', '').replace('"', '')
                    justice += opinion
                    label = line.split("judge")[1].split("filename")[0]
                    label = label.replace(':', '').replace('"', '').replace(",", "").replace(" ", "")
                if text_clean:
                    samples.append(text_cleaner(justice, stopwords))
                else:
                    samples.append(justice)
                labels.append(int(label))
                count += 1
                if count % 100 == 0:
                    print("分词处理进度： " + str(count))
        samples_labels = list(zip(samples, labels))
        random.shuffle(samples_labels)
        samples, labels = zip(*samples_labels)
        with open(samples_path, "w", encoding="utf-8") as fw:
            for sample in samples:
                fw.write(str(sample) + "\n")
        with open(labels_path, "w", encoding="utf-8") as fw:
            for label in labels:
                fw.write(str(label) + "\n")
    else:
        print("从" + samples_path + "和" + labels_path + "中读取数据")
        with open(samples_path, "r", encoding="utf-8") as fr:
            for line in fr:
                samples.append(line.replace("\n", "").replace("'", '"'))
        with open(labels_path, "r", encoding="utf-8") as fr:
            for line in fr:
                labels.append(float(line.replace("\n", "")))
        samples_labels = list(zip(samples, labels))
    samples, labels = zip(*samples_labels)
    return samples, labels


import sys
import re
import random


def predict_label(str):
    return random.sample(range(1,10), 3)


def main():
    predict_labels = []
    for line in open(input_path, 'r', encoding='utf-8'):
        line_json = eval(line)

        labels = predict_label(line_json['features_content'])
        predict_labels.append({"testid":line_json['testid'], "labels_index":labels})

    with open(save_path, "w", encoding="utf-8") as fw:
        for line in predict_labels:
            fw.write(str(line)+"\n")


if __name__ == '__main__':
    input_path = sys.argv[1]
    save_path = sys.argv[2]
    main()
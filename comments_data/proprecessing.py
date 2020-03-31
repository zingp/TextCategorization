import os
import re
import emoji
import numpy as np
import pickle as pkl
from copy import deepcopy
        
        
def clean_text_zh(text):
    """中文数据清洗"""
    # 去除空格
    text = re.sub(' ', '', text)
    # 去掉全角空白符，\u3000 是全角的空白符
    text = re.sub('\u3000', '', text)
    # 去掉 \xa0 是不间断空白符 &nbsp;
    text = re.sub('\xa0', '', text)
    return text

# 清除emoji
def filter_emoji(srcstr, restr=''):  
    """过滤emoji"""
    # 编译匹配表情的正则
    prog = emoji.get_emoji_regexp()
    return prog.sub(restr, srcstr) 

def load_emoji(emoji_file):
    """加载表情和对应的中文"""
    dic = {}
    with open(emoji_file, "r") as f:
        for line in f:
            if len(line.strip("\n").strip()) == 0:
                continue
            line = line.strip("\n")
            line_li = line.split()
            key = line_li[0]
            value = line_li[-1]
            dic[key] = value
    return dic

def emoji2zh(text, emoji_dic):
    """表情替换为中文"""
    prog = emoji.get_emoji_regexp()
    li = re.findall(prog, text)
    for emo in li:
        text = text.replace(text, emoji_dic.get(emo, "表情")) 
    return text


def load_data(filename, emoji_dic):
    data_li = []
    with open(filename, "r") as f:
        n = 0
        for line in f:
            line_li = line.split("\t")
            label = line_li[-1].strip("\n")
            text = line_li[0].strip()
            text = clean_text_zh(text)
            text = emoji2zh(text, emoji_dic)
            data_li.append((text, label))
    return data_li

def split_dataset(data_li, rate=0.8):
    """训练集、验证集、测试集的划分"""
    data = np.array(data_li)
    np.random.seed(123)
    np.random.shuffle(data)
    n = len(data)
    n_train = int(n*rate)
    n_dev = int(n*(1-rate)/2)
    
    train_data = data[:n_train]
    dev_data = data[n_train:n_train+n_dev+1]
    test_data = data[n_train+n_dev+1:]
    return train_data, dev_data, test_data

def write_to_file(filename, data):
    """写入文件"""
    with open(filename, "w") as f:
        for item in data:
            string = item[0] + "\t" + item[-1] + "\n"
            f.write(string)
    print("Write to {}.".format(filename))
    
    
if __name__ == "__main__":
    src_data_file = "./corpus.txt"
    train = "./data/train.txt"
    dev = "./data/dev.txt"
    test = "./data/test.txt"
    emoji_unicode_file = "./data/emoji_unicode.txt"
    dir_name = ["./data", "saved_dict"]
    for d in dir_name:
        if not os.path.exists(d):
            os.mkdir(d)
    
    emoji_dic = load_emoji(emoji_unicode_file)
    data = load_data(src_data_file, emoji_dic)
    train_data, dev_data, test_data = split_dataset(data, rate=0.8)
    write_to_file(train, train_data)
    write_to_file(dev, dev_data)
    write_to_file(test, test_data)
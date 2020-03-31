import os
import time
import torch
import pickle as pkl
from importlib import import_module

'''
模型推理
'''
     
PAD = "<PAD>"
UNK = '<UNK>'
pad_size = 32     # 序列长度
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(filename):
    """加载测试数据"""
    data_li = []
    with open(filename, "r") as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            line_li = line.split("\t")
            data_li.append(line_li)
    return data_li

def load_vocab(vocab_path):
    """加载词表"""
    if os.path.exists(vocab_path):
        vocab = pkl.load(open(vocab_path, 'rb'))
        return vocab


def build_bacth(vocab, s, pad_size=32):
    """构建输入模型的数据"""
    token = [vocab.get(x, vocab.get(UNK)) for x in s]
    seq_len = len(token)
    if len(token) < pad_size:
        token.extend([vocab.get(PAD)] * (pad_size - len(token)))
    else:
        token = token[:pad_size]
        seq_len = len(token)
    token = torch.tensor([token]).to(device)
    seq_len = torch.tensor([[seq_len]])
    return (token, seq_len)


def predic(model, x):
    """预测"""
    model.eval()
    with torch.no_grad():
        output = model(x)
        y = torch.max(output.data, -1)[1].cpu().numpy()
    return y

def create_model(model_name, vocab):
    module = import_module('models.' + model_name)
    config = module.Config(dataset, embedding)
    model = module.Model(config)
    model_path = os.path.join(dataset, 
                    "saved_dict/{}.ckpt".format(model_name)) # 模型
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    return model


if __name__ == "__main__":
    rootdir = "./"
    dataset = 'comments_data'                                  # 数据集
    embedding = 'embedding_Weibo.npz'                          # 裁剪后的embedding
    data_path = os.path.join(dataset, "data")
    vocab_path = os.path.join(data_path, "vocab.pkl")          # 词汇表 
    
    log_dir = os.path.join(rootdir, "log")
    trgdata_dir = os.path.join(data_path, "trgdata")
    for d in [log_dir, trgdata_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    yesterday = time.strftime("%Y%m%d", time.localtime(time.time()-86400))
    srcfile = os.path.join(data_path, "srcfile.{}.txt".format(yesterday))
    trgfile = os.path.join(data_path, "comments_label.{}.txt".format(yesterday))

    model_name = "TextCNN"
    vocab = load_vocab(vocab_path)
    test_data = load_data(srcfile)
    model = create_model(model_name, vocab)

    with open(trgfile, "w") as f:
        for data in test_data:
            comment = data[-1].strip("\n").lower()
            x = build_bacth(vocab, comment, pad_size=pad_size)
            ypred = predic(model, x)
            if type(ypred) is list:
                yp = ypred[-1]
            else:
                yp = ypred
            line = "\t".join([data[0], data[-1].strip("\n"), str(yp)])
            f.write(line)
            f.write("\n")
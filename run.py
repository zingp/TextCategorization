# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse

def parse():
    parser = argparse.ArgumentParser(description='Text Classification')
    # 选择模型
    parser.add_argument('--model', 
                        type=str, 
                        required=True, 
                        help='choose a model: TextCNN, TextRNN, \
                        FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
    # embedding层使用预训练词向量还是重新训练，默认预训练
    parser.add_argument('--embedding', 
                        default='pre_trained', 
                        type=str, 
                        help='random or pre_trained')
    # 词级别还是字级别 默认是字级别
    parser.add_argument('--word', 
                        default=False, 
                        type=bool, 
                        help='True for word, False for char')
    # 网络参数初始化方法
    parser.add_argument('--init_method', 
                        default='xavier',
                        type=str,
                        help='xavier or kaiming')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    #dataset = 'data'  # 数据集
    dataset = 'comments_data'  # 数据集
    # 获取命令行参数
    args = parse()
    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    embedding = 'embedding_Weibo.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif
    print("- "*35)
    print("Start train model : {}".format(model_name))

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model, method=args.init_method)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)

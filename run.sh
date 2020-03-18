#!/bin/sh

python run.py --model "TextCNN" --init_method "kaiming"
python run.py --model "TextRNN" --init_method "kaiming" 
python run.py --model "FastText" --init_method "kaiming"
python run.py --model "TextRCNN" --init_method "kaiming"
python run.py --model "TextRNN_Att" --init_method "kaiming"
python run.py --model "DPCNN" --init_method "kaiming"
python run.py --model "Transformer"

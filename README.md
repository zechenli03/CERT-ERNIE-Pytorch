# CERT-ERNIE-Pytorch

This repository contains code for [Hugging Face implementation of BERT](https://github.com/huggingface/transformers) and uses [huggingface's](https://github.com/huggingface/pytorch-transformers) format [ERNIE](https://github.com/PaddlePaddle/ERNIE) converted by [nghuyong2019](https://github.com/nghuyong/ERNIE-Pytorch).

## Getting Started
You can directly download the ernie model [nghuyong2019](https://github.com/nghuyong/ERNIE-Pytorch) have converted **or** directly load by huggingface's transformers  **or**  convert it with the [nghuyong2019's](https://github.com/nghuyong/ERNIE-Pytorch) code.

Firstly, please intall all the package we needed in this task
```pip install -r requirements.txt```

### Contrastive Self-supervised Learning(CSSL) Pretraining

#### Data Augmentation
If the language in your task dataset is English, for each input sentence x in the target task, you could augment it by first using an English-to-German machine translation model to translate x to y, and then using a German-to-English translation model to translate y to x'. The x' is regarded as an augmented sentence of x. Similarly, you could use an English-to-Chinese machine translation model and a Chinese-to-English machine translation model to obtain another augmented sentence x“.

Then, you could save your augmented data into `augmented_data` folder.

#### MoCo Task
We use Momentum Contrast([MoCo](https://arxiv.org/abs/1911.05722)) to implement CSSL.

* Build a new folder called `moco_model` to store your pretrained model with
```mkdir moco_model```
* You need to change the number of negtive samples in line 86 of `MOCO.py`.






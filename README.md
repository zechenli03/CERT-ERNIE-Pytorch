# CERT-ERNIE-Pytorch

This repository contains code for [Hugging Face implementation of BERT](https://github.com/huggingface/transformers) and uses [huggingface's](https://github.com/huggingface/pytorch-transformers) format [ERNIE](https://github.com/PaddlePaddle/ERNIE) converted by [nghuyong2019](https://github.com/nghuyong/ERNIE-Pytorch).

## Getting Started
You can directly download the ernie model [nghuyong2019](https://github.com/nghuyong/ERNIE-Pytorch) have converted **or** directly load by huggingface's transformers  **or**  convert it with the [nghuyong2019's](https://github.com/nghuyong/ERNIE-Pytorch) code.

Firstly, please intall all the package we needed in this task
```pip install -r requirements.txt```




## Contrastive Self-supervised Learning(CSSL) Pretraining

### Data Augmentation
If the language in your task dataset is English, for each input sentence x in the target task, you could augment it by first using an English-to-German machine translation model to translate x to y, and then using a German-to-English translation model to translate y to x'. The x' is regarded as an augmented sentence of x. Similarly, you could use an English-to-Chinese machine translation model and a Chinese-to-English machine translation model to obtain another augmented sentence x“.

Then, you could save your augmented data into `augmented_data` folder.

### MoCo Task
We use Momentum Contrast([MoCo](https://arxiv.org/abs/1911.05722)) to implement CSSL. The steps are as follows.

* Build a new folder called `moco_model` to store your pretrained model with   
```shell
mkdir moco_model
```
* You need to change the number of negtive samples in [line 86 of `MOCO.py`](https://github.com/Ryanro/CERT-ERNIE-Pytorch/blob/master/MOCO.py#L88).   
**Notice: The amount of Augmentated data(negtive samples) must be an integer multiple of [`batch_size`](https://github.com/Ryanro/CERT-ERNIE-Pytorch/blob/master/MOCO.py#L44)**
* Set your own parameters and run [`MOCO.py`](https://github.com/Ryanro/CERT-ERNIE-Pytorch/blob/master/MOCO.py) to implement pretraining process.
```shell
python MOCO.py \
  --lr 0.0001 \
  --batch-size 32 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed \
  --world-size 1 \
  --rank 0 
```
* After training, you can extract encoder_q from the whole model with    
```shell
python trans.py
```


__P.S. If you want to use an encoder other than ERNIE 2.0, you could change the encoder name or path in [line26 ~ line 38 of `builder.py`](https://github.com/Ryanro/CERT-ERNIE-Pytorch/blob/master/moco/builder.py#L26-L38), [line21 of `MOCO.py`](https://github.com/Ryanro/CERT-ERNIE-Pytorch/blob/master/MOCO.py#L221) and [line16 of `trans.py`](https://github.com/Ryanro/CERT-ERNIE-Pytorch/blob/master/trans.py#L16) with any model [huggingface's](https://github.com/huggingface/pytorch-transformers) provided or fits the [huggingface's](https://github.com/huggingface/pytorch-transformers) format.__




## Fine-tune on GLUE tasks
The [General Language Understanding Evaluation (GLUE) benchmark](https://gluebenchmark.com/) is a collection of nine sentence- or sentence-pair language understanding tasks for evaluating and analyzing natural language understanding systems.

Before running any of these GLUE tasks you should download the
[GLUE data](https://gluebenchmark.com/tasks) by running
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
and unpack it to some directory `$GLUE_DIR`.

You may also need to set the two following environment variables:

* `GLUE_DIR`: This should point to the location of the GLUE data.
* `TASK_NAME`: Task name can be one of CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI. 
* `STATE_DICT`: An optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models.

__Example 1: Fine-tuning from MOCO model__

```shell
export GLUE_DIR=./glue_data
export STATE_DICT=./moco_model/moco.p
export TASK_NAME=RTE

python run_glue.py \
    --model_name_or_path nghuyong/ernie-2.0-large-en \
    --state_dict $STATE_DICT \
    --task_name $TASK_NAME \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluate_during_training \
    --per_device_train_batch_size 16 \
    --weight_decay 0 \
    --learning_rate 3e-5 \
    --num_train_epochs 5.0 \
    --save_steps 156 \
    --warmup_steps 78 \
    --logging_steps 39 \
    --eval_steps 39 \
    --seed 33333 \
    --output_dir /tmp/$TASK_NAME/
```

__Example 2: Fine-tuning from ERNIE model__

```shell
export GLUE_DIR=./glue_data
export TASK_NAME=RTE

python run_glue.py \
    --model_name_or_path nghuyong/ernie-2.0-large-en \
    --task_name $TASK_NAME \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluate_during_training \
    --per_device_train_batch_size 16 \
    --weight_decay 0 \
    --learning_rate 3e-5 \
    --num_train_epochs 5.0 \
    --save_steps 156 \
    --warmup_steps 78 \
    --logging_steps 39 \
    --eval_steps 39 \
    --seed 199733 \
    --output_dir /tmp/$TASK_NAME/
```

__Example 3: Fine-tuning from BERT model__

```shell
export GLUE_DIR=/path/to/glue
export TASK_NAME=MRPC

python run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_device_eval_batch_size=8   \
    --per_device_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir /tmp/$TASK_NAME/
```

where task name can be one of CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI.

The dev set results will be present within the text file 'eval_results.txt' in the specified output_dir. In case of MNLI, since there are two separate dev sets, matched and mismatched, there will be a separate output folder called '/tmp/MNLI-MM/' in addition to '/tmp/MNLI/'.

You can take [```run_cert_rte.sh```](https://github.com/Ryanro/CERT-ERNIE-Pytorch/blob/master/run_cert_rte.sh) as an example.





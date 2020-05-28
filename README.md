# Electra_small
Train a small Electra from scratch and fine tune it with GLUE(SST-2) and with ImDb data. After getting the result, compare it with Bert.
## Table of Contents  
- [Train Electra from Scratch](#train-electra-from-scratch)  
- [Fine Tune the trained Electra_small on GLUE tasks for sequence classification](#fine-tune-the-trained-electra_small-on-glue-tasks-for-sequence-classification)  
- [Fine Tune the pretrained Electra_base on imdb unsupervised dataset ](#fine-tune-the-pretrained-electra_base-on-imdb-unsupervised-dataset)  
- [Fine Tune the post pretrained Electra on Imdb sentiment classification dataset](#fine-tune-the-post-pretrained-electra-on-imdb-sentiment-classification-dataset)  
- [Ablation Experiment of Bert](#ablation-experiment-of-bert)  
## Install   
        
        $ git clone https://github.com/LiZongyue/Electra_small.git
        $ cd ~/Electra_small
        $ pip install .
        

## Train Electra from Scratch  
[Train_from_Scratch](https://github.com/LiZongyue/Electra_small/tree/master/examples/train_from_scratch)  
Before running the pre-training example, you should get a file that contains text on which the language model will be trained. A good example of such text is the [WikiText-103 dataset](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/). Download WikiText.raw for the example.  


## Fine Tune the trained Electra_small on GLUE tasks for sequence classification  
[fine-tune_GLUE](https://github.com/LiZongyue/Electra_small/tree/master/examples/fine-tune_GLUE)  
The General Language Understanding Evaluation (GLUE) benchmark is a collection of nine sentence- or sentence-pair language understanding tasks for evaluating and analyzing natural language understanding systems.  

Accuracy of SST-2 classification : 96.87%
 

For more detail about the [dataset](https://gluebenchmark.com/tasks) and how to run the example, click [here](https://github.com/LiZongyue/Electra_small/blob/master/examples/fine-tune_GLUE/README.md)  
## Fine Tune the pretrained Electra_base on imdb unsupervised dataset  
[fine-tune_pretrained_electra](https://github.com/LiZongyue/Electra_small/tree/master/examples/fine-tune_pretrained_electra)  
This example could also be called as post pre-training Electra. The [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) is for fine-tuning the pretrained Electra-Base. In this task, 50k data samples for unsupervised learning will be used. The data directory is `~/aclImdb/train/unsup/` once the dataset is downloaded through [this link](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz).  

## Fine Tune the post pretrained Electra on Imdb sentiment classification dataset  
[fine-tune_Imdb](https://github.com/LiZongyue/Electra_small/tree/master/examples/fine-tune_Imdb)  
In this example, data supplier is as same as __Fine Tune the pretrained Electra_base on imdb unsupervised dataset__. The data directory is `~/aclImdb/train/pos/`, `~/aclImdb/train/neg/`, `~/aclImdb/test/pos/` and `~/aclImdb/test/neg/` once the dataset is downloaded through [this link](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz). Use [Data Generator](https://github.com/LiZongyue/Electra_small/blob/master/src/Electra_small/utils/DataGenerator.py) to generate the wohle dataset which could be used directly by [the script](https://github.com/LiZongyue/Electra_small/blob/master/examples/fine-tune_Imdb/run_fine-tune.py)  
## Ablation Experiment of Bert  
[ablation-Bert](https://github.com/LiZongyue/Electra_small/tree/master/examples/ablation-Bert)  
In this example, there are two scripts running on Bert. First uses the unsupvised dataset of ImDb to fine tune a pre-trained base Bert, second uses the sentiment classification dataset of ImDb to do the Ablation experiment to compare with Electra.

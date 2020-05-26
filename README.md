# Electra_small
Train a small Electra from scratch and fine tune it with GLUE.
## Table of Contents  
- [Train Electra from Scratch](#train-electra-from-scratch)  
- [Fine Tune the trained Electra_small on GLUE tasks for sequence classification](#fine-tune-the-trained-electra_small-on-glue-tasks-for-sequence-classification)  
- [Fine Tune the pretrained Electra_base on imdb unsupervised dataset ](#fine-tune-the-pretrained-electra_base-on-imdb-unsupervised-dataset)  
## Install   
        
        $ git clone https://github.com/LiZongyue/Electra_small.git
        $ cd ~/Electra_small
        $ pip install .
        

## Train Electra from Scratch  
Before running the pre-training example, you should get a file that contains text on which the language model will be trained. A good example of such text is the [WikiText-103 dataset](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/). Download WikiText.raw for the example.  


## Fine Tune the trained Electra_small on GLUE tasks for sequence classification  
The General Language Understanding Evaluation (GLUE) benchmark is a collection of nine sentence- or sentence-pair language understanding tasks for evaluating and analyzing natural language understanding systems.  
 

For more detail about the [dataset](https://gluebenchmark.com/tasks) and how to run the example, click [here](https://github.com/LiZongyue/Electra_small/blob/master/examples/fine-tune_GLUE/README.md)  
## Fine Tune the pretrained Electra_base on imdb unsupervised dataset  
The [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) is for fine-tuning the pretrained Electra-Base. In this task, 50k data samples for unsupervised learning will be used. The data directory is `~/aclImdb/train/unsup/` once the dataset is downloaded through [this link](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz).

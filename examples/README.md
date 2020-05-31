# Examples

## Ablation Experiments

* __Results of pretrained Bert & Electra on Imdb sentiment classification task.__  
The pre-trained Bert-base-uncased and Electra-Discriminator-base are trained by [Huggingface](https://huggingface.co/), In this experiments they are loaded directly and fine-tuning with the Imdb sentiments classification data.  
* __Results of post-pretrained Bert & Electra on Imdb sentiment classification task.__  
Post-pretrained model means that both of Bert-base-uncased and Electra-Discriminator-base will be further pre-trained on the Imdb unsupervised data. In this case, a better result on sentiment classification is expected due to data domain-adaption.


* __Head for downstream task__  
A same head as SST-2 task is re-used in this experiment. For next step, more different heads will be tested and documented.  

* __Results__  

|    Models   | Train loss after<br>training Head for 5 Epochs     | Train loss after<br>training all for 6 Epochs     | Train Accuracy after<br>training Head for 5 Epochs     | Train Accuracy after<br>training all for 6 Epochs     | Validation loss after<br>training Head for 5 Epochs     | Validation loss after<br>training all for 6 Epochs     | Validation Accuracy after<br>training Head for 5 Epochs     | Validation Accuracy after<br>training all for 6 Epochs (best)     |
| ---------- | :-----------:  | :-----------: | ---------- | :-----------:  | :-----------: | ---------- | :-----------:  | :-----------: |
| Bert-base     | 0.6507     | 0.1696     | 0.6707|  0.9344| 0.6411 | 0.3544 | 0.6887| 0.8734|
| Bert-base post-pretrained     | 0.6287     | 0.1629     | 0.6922 | 0.9359 | 0.6144 | 0.3338 | 0.7001 | 0.8778|
| Electra-base     | 0.6891     | 0.2457     | 0.5354|  0.8981| 0.6853 | 0.2727 | 0.5604| 0.8893|
| Electra-base post-pretrained     | 0.6868     | 0.2089     | 0.5463|  0.9134| 0.6813 | 0.3050 | 0.5771| 0.8941|

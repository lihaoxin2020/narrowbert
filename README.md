# NarrowBERT

Code for paper [NarrowBERT: Accelerating Masked Language Model Pretraining and Inference](https://arxiv.org/abs/2301.04761).

## Dependencies

This implementation is mainly based on [Huggingface Transformers](https://huggingface.co/docs/transformers/index) with optimization package [DeepSpeed](https://www.deepspeed.ai/). Please install these packages to get the most efficiency. 

We provide a DeepSpeed example configuration ```./ds_zero2_1gpu.json```, and feel free to use your own. 

## Implementation

Configuration and model implementation could be found within ```./narrowbert```. The code is mainly adapted from BERT provided on Huggingface. 

We provide training scripts for all tasks we mentioned in the paper, but you can also grab the model and train with your own scripts. 

We provide models for MLM pretraining, sequence classification, and for token classification, which are all we need for experiments mentioned in the paper. 

For tokenizer, we reuse BERT tokenizer provided on Huggingface. In all our experiments, we use pretrained tokenizer from ```bert-base-uncased```.


## Pretraining

```./run_mlm_narrowbert.py``` is the script for pretraining, and is adapted from Huggingface exampla ```run_mlm.py```. You can run it easily with command

```
python ./run_mlm_narrowbert.py ./run_mlm_config.json
```

where ```./run_mlm_config.json``` includes hyperparameters that you can tweak on your own. 


## GLUE/IMDB/Amazon Tests

We adapt from Huggingface example ```run_glue.py``` and provide ```./run_glue_narrowbert.py``` with corresponding configurations. To run the script:

```
python ./run_glue_narrowbert.py [config]
```

where you can replace ```[config]``` with ```./config_glue.json``` or ```./config_imdb.json```. Again, you can modify hyperparameters or choose different tasks of GLUE using these json files. 

Note that Amazon2/5 requires some data preprocessing. We provide the script we used ```amazon.py```. To run preprocessing:

```
python ./amazon.py [cache_path] [amazon2_save_path] [amazon5_save_path]
```


## NER Tests

We use ```./run_ner_narrowbert.pt```, which is, again, adapted from Huggingface example ```run_ner.py```. To run it:

```
python ./run_ner_narrowbert.py ./config_ner.json
```

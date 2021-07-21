
# CLNER

The code is for our ACL-IJCNLP 2021 paper: [Improving Named Entity Recognition by External Context Retrieving and Cooperative Learning](https://arxiv.org/abs/2105.03654)

CLNER is a framework for improving the accuracy of NER models through retrieving external contexts, then use the cooperative learning approach to improve the both input views. The code is initially based on [flair version 0.4.3](https://github.com/flairNLP/flair). Then the code is extended with [knwoledge](https://github.com/Alibaba-NLP/MultilangStructureKD) [distillation](https://github.com/Alibaba-NLP/StructuralKD) and [ACE](https://github.com/Alibaba-NLP/ACE) approaches to distill smaller models or achieve SOTA results. The config files in these repos are also applicable to this code.

<!-- 
## Comparison with State-of-the-Art

| Task | Language | Dataset | ACE | Previous best |
| -------------------------------  | ---  | ----------- | ---------------- | ------------- |
| Named Entity Recognition |English | CoNLL 03 (document-level)   |  **94.6** (F1)  | *94.3 [(Yamada et al., 2020)](https://arxiv.org/pdf/2010.01057.pdf)* |
| Named Entity Recognition |German | CoNLL 03 (document-level)   |  **88.3** (F1) | *86.4 [(Yu et al., 2020)](https://arxiv.org/pdf/2005.07150.pdf)* |
| Named Entity Recognition |German | CoNLL 03 (06 Revision) (document-level)   |  **91.7** (F1)   | *90.3 [(Yu et al., 2020)](https://arxiv.org/pdf/2005.07150.pdf)* |
| Named Entity Recognition |Dutch | CoNLL 02 (document-level)   |  **95.7** (F1) | *93.7 [(Yu et al., 2020)](https://arxiv.org/pdf/2005.07150.pdf)* |
| Named Entity Recognition |Spanish | CoNLL 02 (document-level)   |  **95.9** (F1)  | *90.3 [(Yu et al., 2020)](https://arxiv.org/pdf/2005.07150.pdf)* |
| Named Entity Recognition |English | CoNLL 03 (sentence-level)   |  **93.6** (F1)  | *93.5 [(Baevski et al., 2019)](https://arxiv.org/pdf/1903.07785v1.pdf)* |
| Named Entity Recognition |German | CoNLL 03 (sentence-level)   |  **87.0** (F1) | *86.4 [(Yu et al., 2020)](https://arxiv.org/pdf/2005.07150.pdf)* |
| Named Entity Recognition |German | CoNLL 03 (06 Revision) (sentence-level)   |  **90.5** (F1)   | *90.3 [(Yu et al., 2020)](https://arxiv.org/pdf/2005.07150.pdf)* |
| Named Entity Recognition |Dutch | CoNLL 02 (sentence-level)   |  **94.6** (F1) | *93.7 [(Yu et al., 2020)](https://arxiv.org/pdf/2005.07150.pdf)* |
| Named Entity Recognition |Spanish | CoNLL 02 (sentence-level)   |  **91.7** (F1)  | *90.3 [(Yu et al., 2020)](https://arxiv.org/pdf/2005.07150.pdf)* |
| POS Tagging |English | Ritter's |  **93.4** (Acc)  | *90.1 [(Nguyen et al., 2020)](https://arxiv.org/pdf/2005.10200.pdf)* |
| POS Tagging |English | Ark |  **94.4** (Acc)  | *94.1 [(Nguyen et al., 2020)](https://arxiv.org/pdf/2005.10200.pdf)* |
| POS Tagging |English | TweeBank v2 |  **95.8** (Acc)  | *95.2 [(Nguyen et al., 2020)](https://arxiv.org/pdf/2005.10200.pdf)* |
| Aspect Extraction |English | SemEval 2014 Laptop |  **87.4** (F1)  | *84.3 [(Xu et al., 2019)](https://arxiv.org/pdf/1904.02232.pdf)* |
| Aspect Extraction |English | SemEval 2014 Restaurant |  **92.0** (F1)  | *87.1 [(Wei et al., 2020)](https://www.aclweb.org/anthology/2020.acl-main.339/)* |
| Aspect Extraction |English | SemEval 2015 Restaurant |  **80.3** (F1)  | *72.7 [(Wei et al., 2020)](https://www.aclweb.org/anthology/2020.acl-main.339/)* |
| Aspect Extraction |English | SemEval 2016 Restaurant |  **81.3** (F1)  | *78.0 [(Xu et al., 2019)](https://arxiv.org/pdf/1904.02232.pdf)* |
| Dependency Parsing | English | PTB    |  **95.7** (LAS)  | *95.3 [(Wang et al., 2020)](https://arxiv.org/pdf/2010.05003.pdf)* |
| Semantic Dependency Parsing | English | DM ID   |  **95.6** (LF1)  | *94.4 [(Fernández-González and Gómez-Rodríguez, 2020)](https://www.aclweb.org/anthology/2020.acl-main.629/)* |
| Semantic Dependency Parsing | English | DM OOD   |  **92.6** (LF1)  | *91.0 [(Fernández-González and Gómez-Rodríguez, 2020)](https://www.aclweb.org/anthology/2020.acl-main.629/)* |
| Semantic Dependency Parsing | English | PAS ID   |  **95.8** (LF1)  | *95.1 [(Fernández-González and Gómez-Rodríguez, 2020)](https://www.aclweb.org/anthology/2020.acl-main.629/)* |
| Semantic Dependency Parsing | English | PAS OOD   |  **94.6** (LF1)  | *93.4 [(Fernández-González and Gómez-Rodríguez, 2020)](https://www.aclweb.org/anthology/2020.acl-main.629/)* |
| Semantic Dependency Parsing | English | PSD ID   |  **83.8** (LF1)  | *82.6 [(Fernández-González and Gómez-Rodríguez, 2020)](https://www.aclweb.org/anthology/2020.acl-main.629/)* |
| Semantic Dependency Parsing | English | PSD OOD   |  **83.4** (LF1)  | *82.0 [(Fernández-González and Gómez-Rodríguez, 2020)](https://www.aclweb.org/anthology/2020.acl-main.629/)* |
 -->
## Guide

- [Requirements](#requirements)
- [Datasets](#datasets)
<!-- - [Pretrained Models](#Pretrained-Models)
  - [Instructions for Reproducing Results](#Instructions-for-Reproducing-Results) -->
<!-- - [Download Embeddings](#Download-Embeddings) -->
  <!-- - [Dump Fine-tuned Embeddings in the Pretrained Models ](#Dump-Fine-tuned-Embeddings-in-the-Pretrained-Models) -->
- [Training](#training)
<!--   - [Training ACE Models](#training-ace-models)
  - [Train on Your Own Dataset](#Train-on-Your-Own-Dataset)
  - [Set the Embeddings](#Set-the-Embeddings)
  - [(Optional) Fine-tune Transformer-based Embeddings](#Optional-Fine-tune-Transformer-based-Embeddings)
  - [(Optional) Extract Document Features for BERT Embeddings](#Optional-Extract-Document-Features-for-BERT-Embeddings) -->
- [Parse files](#parse-files)
- [Config File](#Config-File)
<!-- - [TODO](#todo) -->
- [Citing Us](#Citing-Us)
- [Contact](#contact)

## Requirements
The project is based on PyTorch 1.1+ and Python 3.6+. To run our code, install:

```
pip install -r requirements.txt
```

The following requirements should be satisfied:
* [transformers](https://github.com/huggingface/transformers): **3.0.0** 

## Datasets
The datasets used in our paper is available [here](https://1drv.ms/u/s!Am53YNAPSsodg9ce3ovPukuFtSj6NQ?e=tpCvf8).

## Training

### Training NER Models with External Contexts

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/wnut17_doc.yaml
```

### Training NER Models with Cooperative Learning

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/wnut17_doc_cl_kl.yaml
CUDA_VISIBLE_DEVICES=0 python train.py --config config/wnut17_doc_cl_l2.yaml
```

### Train on Your Own Dataset

To set the dataset manully, you can set the dataset in the `$confile_file` by:

```yaml
targets: ner
ner:
  Corpus: ColumnCorpus-1
  ColumnCorpus-1: 
    data_folder: datasets/conll_03_english
    column_format:
      0: text
      1: pos
      2: chunk
      3: ner
    tag_to_bioes: ner
  tag_dictionary: resources/taggers/your_ner_tags.pkl
```


The `tag_dictionary` is a path to the tag dictionary for the task. If the path does not exist, the code will generate a tag dictionary at the path automatically. The dataset format is: `Corpus: $CorpusClassName-$id`, where `$id` is the name of datasets (anything you like). You can train multiple datasets jointly. For example:

Please refer to [Config File](#Config-File) for more details.

## Parse files

If you want to parse a certain file, add `train` in the file name and put the file in a certain `$dir` (for example, `parse_file_dir/train.your_file_name`). Run:

```
CUDA_VISIBLE_DEVICES=0 python train.py --config $config_file --parse --target_dir $dir --keep_order
```

The format of the file should be `column_format={0: 'text', 1:'ner'}` for sequence labeling or you can modifiy line 232 in `train.py`. The parsed results will be in `outputs/`.
Note that you may need to preprocess your file with the dummy tags for prediction, please check this [issue](https://github.com/Alibaba-NLP/ACE/issues/12) for more details.

## Config File

The config files are based on yaml format.

* `targets`: The target task
  * `ner`: named entity recognition
  * `upos`: part-of-speech tagging
  * `chunk`: chunking
  * `ast`: abstract extraction
  * `dependency`: dependency parsing
  * `enhancedud`: semantic dependency parsing/enhanced universal dependency parsing
* `ner`: An example for the `targets`. If `targets: ner`, then the code will read the values with the key of `ner`.
  * `Corpus`: The training corpora for the model, use `:` to split different corpora.
  * `tag_dictionary`: A path to the tag dictionary for the task. If the path does not exist, the code will generate a tag dictionary at the path automatically.
* `target_dir`: Save directory.
* `model_name`: The trained models will be save in `$target_dir/$model_name`.
* `model`: The model to train, depending on the task.
  * `FastSequenceTagger`: Sequence labeling model. The values are the parameters.
  * `SemanticDependencyParser`: Syntactic/semantic dependency parsing model. The values are the parameters.
* `embeddings`: The embeddings for the model, each key is the class name of the embedding and the values of the key are the parameters, see `flair/embeddings.py` for more details. For each embedding, use `$classname-$id` to represent the class. For example, if you want to use BERT and M-BERT for a single model, you can name: `TransformerWordEmbeddings-0`, `TransformerWordEmbeddings-1`.
* `trainer`: The trainer class.
  * `ModelFinetuner`: The trainer for fine-tuning embeddings or simply train a task model without ACE.
  * `ReinforcementTrainer`: The trainer for training ACE.
* `train`: the parameters for the `train` function in `trainer` (for example, `ReinforcementTrainer.train()`).

## TODO
* Knowledge Distillation with ACE: [Wang et al., 2020](https://github.com/Alibaba-NLP/MultilangStructureKD)

## Citing Us
If you feel the code helpful, please cite:
```
@inproceedings{wang2021improving,
    title = "{{Improving Named Entity Recognition by External Context Retrieving and Cooperative Learning}}",
    author={Wang, Xinyu and Jiang, Yong and Bach, Nguyen and Wang, Tao and Huang, Zhongqiang and Huang, Fei and Tu, Kewei},
    booktitle = "{the Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (\textbf{ACL-IJCNLP 2021})}",
    month = aug,
    year = "2021",
    publisher = "Association for Computational Linguistics",
}
```

## Contact 

Feel free to email your questions or comments to issues or to [Xinyu Wang](http://wangxinyu0922.github.io/).


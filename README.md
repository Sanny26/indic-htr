# Indic-HTR
Handwriting recognition for various Indic scripts using deep learning(STN, CNN, LSTM)

For more info about the pipeline, please refer to *IIIT-INDIC-HW-WORDS: A Dataset for Indic Handwritten Text Recognition*

[\[Paper\]](http://cvit.iiit.ac.in/images/ConferencePapers/2021/iiit-indic-hw-words.pdf)| [\[Dataset\]](http://cvit.iiit.ac.in/research/projects/cvit-projects/iiit-indic-hw-words) | [\[Teaser\]](http://cvit.iiit.ac.in/images/Projects/iiit-indic-hw-words/331.mp4)

## Training and Evaluation:

### Setting up the environment:
We are using Poetry, a Python dependency manager for the repository:

- Install instructions for Poetry are mentioned here

- Once installed, clone the Indic-HTR repo:
```
cd indic-htr
poetry install      # install the dependencies from poetry.lock file
poetry shell        # activate the virutal environment for this repo
```

### Dataset preparation:
- Create LMDB files for train, validation and test splits.
```
python3 tools/create_dataset.py --root_dir <dataset-dir-path> --folder <split> --save_as <output-lmdb-path>
```

## Training:

To train model(TPS-ResNet-BiLSTM-CTC) from scratch:
```
python lang_train.py --mode train --lang <lang_code> --trainRoot <train_lmdb_path> --valRoot <val_lmdb_path> --cuda
```
Refer to *lang_train.py* and *config.py* for default settings and additional parameter settings.

Language codes for Indic scripts:
|Bengali|Gujarati|Gurumukhi|Odia|Kannada|Malayalam|Tamil|Urdu|
|-------|--------|---------|----|-------|---------|-----|----|
|   bn  |   gu   |    pn   | od |   kn  |    ma   |  ta | ur |

## Evaluation and testing:
To generate predictions for a <test-lmdb> file, try:
```
python lang_train.py --lang <lang_code> --mode test --val_dir <test-lmdb-path> --output <save-predictions-path>
```

python lang_train.py --lang ta --mode test --valRoot "/media/san/New Volume/qrmary/datasets/ta/ta-test-lmdb" --out ta_preds.txt --pretrained "/media/san/New Volume/Ada_share3_backup/imprint/ta_final/best_cer.pth"


To evaluate the generated predictions, try the following:
```
python tools/score.py --preds <save-predictions-path>
```
or
```
python tools/oov_score.py --preds <save-predictions-path> --vocab <path-to-train-vocab>
```
to get WER and CER for OOV words only.

Please check the code for more config options.

## Points to note:

- The *alphabet* folder contains text file per language. Each text lists all unique Unicode symbols/characters used in the IIIT-INDIC-HW words dataset.
Please 


python lang_train.py --lang ur --mode test --valRoot "../ur_test_lmdb/" --out tmp/ur_preds.txt --pretrained "/media/san/New Volume/Ada_share3_backup/imprint/ur_iam/best_cer.pth"
python lang_train.py --lang ur --mode test --valRoot "../ur_test_lmdb/" --out ur_preds.txt --pretrained "/media/san/New Volume/Ada_share3_backup/imprint/ur_iam/best_cer.pth"
python tools/score.py --preds ur_preds.txt
python tools/oov_score.py --preds ur_preds.txt --tr_vocab "/media/san/New Volume/qrmary/vocab/train_ur.txt"


## Pretrained Models:
Will be released soon.

## Acknowledgment:
We thank these repositories [crnn.pytorch](https://github.com/meijieru/crnn.pytorch), [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) for releasing their codes.
Code structure is inspired from [aster.pytorch](https://github.com/ayumiymk/aster.pytorch).

## Citation:
Please consider citing this work in your publications if it helps your research.
```
@InProceedings{10.1007/978-3-030-86337-1_30,
author="Gongidi, Santhoshini
and Jawahar, C. V.",
title="iiit-indic-hw-words: A Dataset for Indic Handwritten Text Recognition",
booktitle="Document Analysis and Recognition -- ICDAR 2021",
year="2021",
}
```

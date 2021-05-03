# SLU-M-BERT

## Preprocessing

### To preprocess the English dataset

```
mkdir -p memory/enfr/en
cd preprocess
python3 preprocess_enfr.py --lang=en
```
This will save `train*.npy`, `dev*.npy`, and `test*.npy` 
in the `memory` directory.

### To preprocess the Chinese dataset

```
mkdir -p memory/cn
cd preprocess
python3 preprocess_cn.py
```



## Training and testing

### Train/test a model for English dataset

```
python3 baseline/train.py --epochs=50 --lr=1e-3 --dir=memory/enfr/en \
--lm_epochs=150 --epochs=20 --batch_size=128 --num_filters=128 \
--num_lstm_layers=1 --lstm_hidden=128 --embed_dim=128 \
--weight_tying=1 --save_model_path=best_model/en_best.pt
```

This gives best validation accuracy of 67.02%.

To evaluate on the test set,

```
python3 baseline/eval.py --test_dir=memory/enfr/en --model=best_model/en_best.pt
```


### Train/test a model for Chinese dataset

```
python3 baseline/train.py --epochs=50 --lr=1e-3 --dir=memory/cn/ \
--lm_epochs=0 --epochs=30 --batch_size=64 --num_filters=128 \
--num_lstm_layers=1 --lstm_hidden=128 --embed_dim=128 \
--weight_tying=0 --save_model_path=best_model/cn_best.pt
```

This gives best validation accuracy of 60.29%.

Test labels for CN dataset are not given.


### Baseline Experiment results

Experiment results are in `expr.log` and scripts for experiments are expr*.sh


## Roberta Language Model

### Preprocess unlabelled CN large dataset for Masked LM

```
python3 roberta/preprocess_cn_mlm.py --dir=Datasets/chinese_LM_data --outpath=memory/cn_lm.npy
```

### Pretrain our roberta LM using CN large dataset

```
python3 roberta/train_lm.py --data=memory/cn_lm.npy --tokenizer=tokenizer/ipa_tokenizer.json --output_dir=roberta/logs
```

### Preprocess labelled CN dataset for finetuning

```
python3 roberta/preprocess_cn.py --data_dir=Datasets/catslu_v2/preprocessed/audio/ --memory_dir=memory/roberta/
```

### Finetune

```
python3 roberta/finetune.py --data_dir=memory/roberta/  --pretrained=roberta/logs --tokenizer=tokenizer/ipa_tokenizer.json \
                   --save_model_path=best_model/roberta.pt --scheduler=1
```

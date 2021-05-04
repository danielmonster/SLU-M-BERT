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
python3 baseline/train.py --epochs=20 --lr=1e-3 --dir=memory/enfr/en \
--lm_epochs=150  --batch_size=128 --num_filters=128 \
--num_lstm_layers=1 --lstm_hidden=128 --embed_dim=128 \
--weight_tying=1 --save_model_path=best_model/en_best.pt
```

This gives best validation accuracy of 67.57%.

To evaluate on the test set,

```
python3 baseline/eval.py --test_dir=memory/enfr/en --model=best_model/en_best.pt
```


### Train/test a model for Chinese dataset

```
python3 baseline/train.py --epochs=30 --lr=1e-3 --dir=memory/cn/ \
--lm_epochs=0 --batch_size=64 --num_filters=128 \
--num_lstm_layers=1 --lstm_hidden=128 --embed_dim=128 \
--weight_tying=0 --save_model_path=best_model/cn_best.pt
```

This gives best validation accuracy of 60.82%.

Test labels for CN dataset are not given.




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
mkdir -p memory/roberta/cn
python3 roberta/preprocess_cn.py --data_dir=Datasets/catslu_v2/preprocessed/audio/ --memory_dir=memory/roberta/cn
```

### Finetune on CN

```
python3 roberta/finetune.py --data_dir=memory/roberta/cn  --tokenizer=tokenizer/ipa_tokenizer.json \
                    --save_model_path=best_model/roberta_cn.pt --scheduler=1
```



### Preprocess labelled EN dataset for training

```
mkdir -p memory/roberta/en
python3 roberta/preprocess_en.py --data_dir=Datasets/smart-devices-en-fr/ \
        --memory_dir=memory/roberta/en
```

### Directly train on EN (without pretraining)

```
python3 roberta/finetune.py --data_dir=memory/roberta/en  --tokenizer=tokenizer/ipa_tokenizer.json \
                    --save_model_path=best_model/roberta_en.pt --scheduler=1
```


### Ablation studies for Roberta

```
rm -rf expr.log
bash roberta/expr_en.sh
bash roberta/expr_cn.sh
```

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
--lm_epochs=0 --batch_size=128 --num_filters=128 \
--num_lstm_layers=1 --lstm_hidden=128 --embed_dim=128 \
--weight_tying=1 --save_model_path=best_model/en_best.pt
```
The above command sets `-lm_epochs` to 0, so no LM will be done.
If you want to do pre-training with LM and then fine-tuning, please set the following arguments

```
--lm_epochs=30 --lm_dir=Datasets/en_LM_data --save_model_path=best_model/en_lm_best.pt
```

To evaluate on the test set,

```
python3 baseline/eval.py --test_dir=memory/enfr/en --model=best_model/en_best.pt
```


### Train/test a model for Chinese dataset

```
python3 baseline/train.py --epochs=20 --lr=1e-3 --dir=memory/cn/ \
--lm_epochs=0 --batch_size=64 --num_filters=128 \
--num_lstm_layers=1 --lstm_hidden=128 --embed_dim=128 \
--weight_tying=1 --save_model_path=best_model/cn_best.pt
```

Or with lm:
```
--lm_epochs=30 --lm_dir=Datasets/chinese_LM_data --save_model_path=best_model/cn_lm_best.pt
```


To evaluate on the test set,

```
python3 baseline/eval.py --test_dir=memory/cn --model=best_model/cn_best.pt
```



## Roberta Language Model

Below we describe the training pipeline of RoBERTa on our datasets.
First we desciribe the steps for the Chinese dataset.
We first pre-train our RoBERTa using a large un-labelled Chinese corpus from `Datasets/chinese_LM_data/`.
We then fine-tune RoBERTa for intent classification on a labelled Chinese dataset from `Datasets/smart-devices-en-fr/`.
We also provide a way to directly train RoBERTa for intent classification without any pre-training.

### Preprocess unlabelled CN LM dataset for Masked LM

```
python3 roberta/preprocess_cn_mlm.py --dir=Datasets/chinese_LM_data --outpath=memory/cn_lm.npy
```

### Preprocess labelled CN dataset for finetuning

```
mkdir -p memory/roberta/cn
python3 roberta/preprocess_cn.py --data_dir=Datasets/catslu_v2/preprocessed/audio/ --memory_dir=memory/roberta/cn
```


### Pretrain RoBERTa with MLM

```
mkdir -p roberta/logs
python3 roberta/train_lm.py --data=memory/cn_lm.npy --tokenizer=tokenizer/ipa_tokenizer.json --output_dir=roberta/logs
```

### Intent classification using pretrained RoBERTa (Finetuning)

```
python3 roberta/finetune.py --data_dir=memory/roberta/cn  --pretrained=roberta/logs --tokenizer=tokenizer/ipa_tokenizer.json \
                   --save_model_path=best_model/roberta_cn.pt --scheduler=1
```

### Intent classification directly (no pretraining)

```
python3 roberta/finetune.py --data_dir=memory/roberta/cn  --tokenizer=tokenizer/ipa_tokenizer.json \
                    --save_model_path=best_model/roberta_cn.pt --scheduler=1
```




The similar training steps for English dataset are also provided below.



### Preprocess unlabelled EN LM dataset for Masked LM

```
python3 roberta/preprocess_en_mlm.py --dir=Datasets/en_LM_data --outpath=memory/en_lm.npy
```


### Preprocess labelled EN dataset for training

```
mkdir -p memory/roberta/en
python3 roberta/preprocess_en.py --data_dir=Datasets/smart-devices-en-fr/ \
        --memory_dir=memory/roberta/en
```

### Pretrain RoBERTa with MLM

```
mkdir -p roberta/logs-en
python3 roberta/train_lm.py --data=memory/en_lm.npy --tokenizer=tokenizer/ipa_tokenizer.json --output_dir=roberta/logs-en
```

### Intent classification using pretrained RoBERTa (Finetuning)

```
python3 roberta/finetune.py --data_dir=memory/roberta/en  --pretrained=roberta/logs-en --tokenizer=tokenizer/ipa_tokenizer.json \
                   --save_model_path=best_model/roberta_en.pt --scheduler=1
```

### Intent classification directly (no pretraining)

```
python3 roberta/finetune.py --data_dir=memory/roberta/en  --tokenizer=tokenizer/ipa_tokenizer.json \
                    --save_model_path=best_model/roberta_en.pt --scheduler=1
```


To evaluate on the test set,

```
python3 roberta/eval.py --test_dir=memory/roberta/cn --tokenizer=tokenizer/ipa_tokenizer.json --model=best_model/roberta_cn.pt
python3 roberta/eval.py --test_dir=memory/roberta/en --tokenizer=tokenizer/ipa_tokenizer.json --model=best_model/roberta_en.pt
```



### Ablation studies for Roberta

```
rm -rf expr.log
bash roberta/expr_en.sh
bash roberta/expr_cn.sh
```

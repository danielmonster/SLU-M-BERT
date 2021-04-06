# SLU-M-BERT

## Preprocessing

### To preprocess the English dataset

```
mkdir memory
cd preprocess
python3 preprocess/preprocess_enfr.py --lang=en
```
This will save `train*.npy`, `dev*.npy`, and `test*.npy` 
in the `memory` directory.

Similar steps are done for Chinese dataset 
in `preprocess/preprocess_data_Chinese.ipynb`.


## Training and testing

### Train/test a model for English dataset

```
python3 baseline/train.py --epochs=50 --lr=1e-4 --dir=memory/enfr/en \
--save_model_path=best_model/en_best.pt
```

This gives best validation accuracy of 60.03%.

To evaluate on the test set,

```
python3 baseline/eval.py --test_dir=memory/enfr/en --model=best_model/en_best.pt
```


### Train/test a model for Chinese dataset

```
python3 baseline/train.py --epochs=50 --lr=1e-3 --dir=memory/cn/ \
--save_model_path=best_model/cn_best.pt
```

This gives best validation accuracy of 68.12%.

Test labels for CN dataset are not given.



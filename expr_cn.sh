#!/bin/bash



echo "Testing Chinese dataset" >> expr.log
echo "" >> expr.log

for i in {0,100}
do
    echo "lm_epochs = $i" >> expr.log
    python3 baseline/train.py --epochs=50 --lr=1e-3 --dir=memory/cn \
    --lm_epochs=$i --epochs=20 --batch_size=128 --num_filters=128 \
    --num_lstm_layers=1 --lstm_hidden=128 --embed_dim=128 \
    --weight_tying=1 --save_model_path=best_model/en_best.pt --verbose=0 >> expr.log
done

echo "" >> expr.log
echo "" >> expr.log

for i in {64,128,256,512}
do
    echo "num_filters = $i" >> expr.log
    python3 baseline/train.py --epochs=50 --lr=1e-3 --dir=memory/cn \
    --lm_epochs=150 --epochs=20 --batch_size=128 --num_filters=$i \
    --num_lstm_layers=1 --lstm_hidden=128 --embed_dim=128 \
    --weight_tying=1 --save_model_path=best_model/en_best.pt --verbose=0 >> expr.log
done

echo "" >> expr.log
echo "" >> expr.log


for i in {128,256,512,1024}
do
    echo "lstm_hidden = $i" >> expr.log
    python3 baseline/train.py --epochs=50 --lr=1e-3 --dir=memory/cn \
    --lm_epochs=150 --epochs=20 --batch_size=128 --num_filters=128 \
    --num_lstm_layers=1 --lstm_hidden=$i --embed_dim=128 \
    --weight_tying=1 --save_model_path=best_model/en_best.pt --verbose=0 >> expr.log
done


echo "" >> expr.log
echo "" >> expr.log

for i in {1,2,3}
do
    echo "num_lstm_layers = $i" >> expr.log
    python3 baseline/train.py --epochs=50 --lr=1e-3 --dir=memory/cn \
    --lm_epochs=150 --epochs=20 --batch_size=128 --num_filters=128 \
    --num_lstm_layers=$i --lstm_hidden=128 --embed_dim=128 \
    --weight_tying=1 --save_model_path=best_model/en_best.pt --verbose=0 >> expr.log
done


echo "" >> expr.log
echo "" >> expr.log


for i in {128,512}
do
    echo "embed_dim = $i" >> expr.log
    python3 baseline/train.py --epochs=50 --lr=1e-3 --dir=memory/cn \
    --lm_epochs=150 --epochs=20 --batch_size=128 --num_filters=128 \
    --num_lstm_layers=1 --lstm_hidden=128 --embed_dim=$i \
    --weight_tying=1 --save_model_path=best_model/en_best.pt --verbose=0 >> expr.log
done


echo "" >> expr.log
echo "" >> expr.log

#!/bin/bash


echo "Testing Chinese dataset with Roberta" >> expr.log
echo "" >> expr.log


for i in {1, 6, 12, 24}
do
    echo "heads = $i" >> expr.log
    python3 roberta/finetune.py --data_dir=memory/roberta/cn  --tokenizer=tokenizer/ipa_tokenizer.json \
        --heads=$i --save_model_path=best_model/roberta_cn.pt --scheduler=1 >> expr.log
done

echo "" >> expr.log
echo "" >> expr.log


for i in {1, 3, 6, 12}
do
    echo "num_layers= $i" >> expr.log
    python3 roberta/finetune.py --data_dir=memory/roberta/cn  --tokenizer=tokenizer/ipa_tokenizer.json \
        --num_layer$i --save_model_path=best_model/roberta_cn.pt --scheduler=1 >> expr.log
done


echo "" >> expr.log
echo "" >> expr.log

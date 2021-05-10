import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaForSequenceClassification, AdamW, PreTrainedTokenizerFast, RobertaConfig
import os
from sklearn.metrics import f1_score, accuracy_score, classification_report
from finetune import PhoneRobertaDataset

import argparse

# python3 roberta/eval.py --test_dir=memory/roberta/cn --tokenizer=tokenizer/ipa_tokenizer.json --model=best_model/roberta_cn.pt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--num_layers', type=int, default=6)
    return parser.parse_args()



def report_acc_f1(y_pred, y_true):
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return f1_macro, f1_micro, accuracy, report




def evaluate(model, device, test_loader):
    model.to(device)
    model.eval()
    preds_all = []
    labels_all = []
    with torch.no_grad():
    
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(test_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            
            _, predicted = torch.max(outputs.logits, 1)
            preds_all += predicted.cpu().numpy().tolist()
            labels_all += labels.cpu().numpy().tolist() 
    
    f1_macro, f1_micro, acc, report = report_acc_f1(preds_all, labels_all)
    print("F1 Macro: {}, F1 Micro: {}".format(f1_macro, f1_micro))
    print("Accuracy: {}".format(acc))
    print(report)
    

def main(args):
    test_x = np.load(os.path.join(args.test_dir, "test_x.npy"), allow_pickle=True)
    test_y = np.load(os.path.join(args.test_dir, "test_y.npy"), allow_pickle=True)
    num_classes = len(np.unique(test_y))
    
    tokenizer_path = args.tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path, max_len=512, mask_token="<mask>", pad_token="<pad>")
    test_dataset = PhoneRobertaDataset(test_x, test_y, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



    config = RobertaConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=514,
        num_attention_heads=args.heads, # default 12
        num_hidden_layers=args.num_layers, # default 6
        type_vocab_size=1,
        num_labels=num_classes
         )
    model = RobertaForSequenceClassification(config)  
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(args.model))
    evaluate(model, device, test_loader)

if __name__ == '__main__':
    args = get_args()
    main(args)

from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, AdamW, PreTrainedTokenizerFast, RobertaConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
from torch.utils.data import Dataset
import torch
import numpy as np
import argparse
import os
import time
from transformers import AdamW
from sklearn.metrics import f1_score, accuracy_score, classification_report


# python3 roberta/finetune.py --data_dir=memory/roberta/cn  --pretrained=roberta/logs --tokenizer=tokenizer/ipa_tokenizer.json \
#                    --save_model_path=best_model/roberta_cn.pt --scheduler=1

# python3 roberta/finetune.py --data_dir=memory/roberta/cn  --tokenizer=tokenizer/ipa_tokenizer.json \
#                    --save_model_path=best_model/roberta_cn.pt --scheduler=1

# python3 roberta/finetune.py --data_dir=memory/roberta/en  --tokenizer=tokenizer/ipa_tokenizer.json \
#                    --save_model_path=best_model/roberta_en.pt --scheduler=1

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



# Reproducibility
torch.manual_seed(11785)
np.random.seed(11785)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--scheduler', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--save_model_path', type=str, default=None)

    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--num_layers', type=int, default=6)
    return parser.parse_args()




class PhoneRobertaDataset(Dataset):
    def __init__(self, data, labels, tokenizer):
        self.tokenizer = tokenizer
        x = list(data)
        self.encoding = tokenizer(x, return_tensors='pt', padding=True, truncation=True)
        self.input_ids = self.encoding['input_ids']
        self.attention_mask = self.encoding['attention_mask']
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return self.input_ids[i], self.attention_mask[i], self.labels[i]


def report_acc_f1(y_pred, y_true):
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return f1_macro, f1_micro, accuracy, report



def train_epoch(model, train_loader, optimizer, verbose):
    model.train()
    running_loss = 0.0
    total_predictions = 0
    correct_predictions = 0
    
    start_time = time.time()
    for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        
        loss = F.cross_entropy(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.logits, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        running_loss += loss.item()

    
    end_time = time.time()
    running_loss /= len(train_loader)
    accuracy = (correct_predictions/total_predictions) * 100.0
    if verbose:
        print("Training loss: ", running_loss, "Time: ", end_time - start_time, 's')
        print("Training Accuracy", accuracy, "%")
    return running_loss, accuracy


def valid_epoch(model, valid_loader, verbose, print_report=False):
    start_time = time.time()
    preds_all = []
    labels_all = []
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        total_predictions = 0
        correct_predictions = 0
    
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(valid_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
  
            loss = F.cross_entropy(outputs.logits, labels)
            _, predicted = torch.max(outputs.logits, 1)
            preds_all += predicted.cpu().numpy().tolist()
            labels_all += labels.cpu().numpy().tolist() 
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            running_loss += loss.item()
    
    end_time = time.time()
    running_loss /= len(valid_loader)
    accuracy = (correct_predictions/total_predictions) * 100.0
    if verbose:
        print("Validation loss: ", running_loss, "Time: ", end_time - start_time, 's')
    
    f1_macro, f1_micro, acc, report = report_acc_f1(preds_all, labels_all)
    print("F1 Macro: {}, F1 Micro: {}".format(f1_macro, f1_micro))
    print("Accuracy: {}".format(acc))
    
    if print_report:
        print(report)

    return running_loss, accuracy




def main(args):
    train_x = np.load(os.path.join(args.data_dir, "train_x.npy"), allow_pickle=True)
    train_y = np.load(os.path.join(args.data_dir, "train_y.npy"), allow_pickle=True)
    valid_x = np.load(os.path.join(args.data_dir, "dev_x.npy"), allow_pickle=True)
    valid_y = np.load(os.path.join(args.data_dir, "dev_y.npy"), allow_pickle=True)
    
    tokenizer_path = args.tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path, max_len=512, mask_token="<mask>", pad_token="<pad>")
    train_dataset = PhoneRobertaDataset(train_x, train_y, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = PhoneRobertaDataset(valid_x, valid_y, tokenizer)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    num_classes = len(np.unique(train_y))
    
    lr = args.lr
    num_epochs = args.epochs
    verbose = args.verbose
    
    if args.pretrained:
        model = RobertaForSequenceClassification.from_pretrained(args.pretrained, num_labels=num_classes)
    else:
        config = RobertaConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=514,
        num_attention_heads=args.heads, # default 12
        num_hidden_layers=args.num_layers, # default 6
        type_vocab_size=1,
        num_labels=num_classes
         )
        model = RobertaForSequenceClassification(config)
        
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, 
                                    verbose=verbose)
    
    best_model_dict = None
    best_acc = 0
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, verbose)
        valid_loss, valid_acc = valid_epoch(model, valid_loader, verbose)
        if verbose:
            print("Epoch {} finished.".format(epoch))
            print('='*20)
        if args.scheduler:
            scheduler.step(valid_loss)
        
        if valid_acc > best_acc:
            best_model_dict = model.state_dict()
            best_acc = valid_acc
            
    if best_model_dict and args.save_model_path:
        torch.save(best_model_dict, args.save_model_path)
    
    model.load_state_dict(best_model_dict)
    print("Evaluate on validation using the best model")
    valid_epoch(model, valid_loader, verbose, print_report=True)
    print("Best validation accuracy: ", best_acc, "%")


if __name__ == "__main__":
    args = get_args()
    main(args)
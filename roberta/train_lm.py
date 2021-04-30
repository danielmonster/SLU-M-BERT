import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import Trainer, TrainingArguments
from tokenizers.processors import BertProcessing
import argparse


# python3 roberta/train_lm.py --data=memory/cn_lm.npy --tokenizer=tokenizer/ipa_tokenizer.json --output_dir=roberta/logs

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    return parser.parse_args()


class PhoneDatasetMLM(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        # tokenizer.enable_truncation(max_length=512)
        # or use the RobertaTokenizer from `transformers` directly.
        self.x = [tokenizer.encode(x) for x in data]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.x[i])


def main(args):
    data = np.load(args.data, allow_pickle=True)
    tokenizer_path = args.tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path, max_len=512, mask_token="<mask>", pad_token="<pad>")
    tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.convert_tokens_to_ids("</s>")),
            ("<s>", tokenizer.convert_tokens_to_ids("<s>")),
        )

    config = RobertaConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )
    

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    dataset = PhoneDatasetMLM(data, tokenizer)

    model = RobertaForMaskedLM(config=config)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=64,
        logging_steps=2,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    args = get_args()
    main(args)

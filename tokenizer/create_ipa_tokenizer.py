import random
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import argparse



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="./")
    return parser.parse_args()




def main(args):
    # copy from https://github.com/xinjli/allosaurus
    ipa0 = ['I', 'a', 'aː', 'ã', 'ă', 'b', 'bʲ', 'bʲj', 'bʷ', 'bʼ', 'bː', 'b̞', 'b̤', 'b̥', 'c', 'd', 'dʒ', 'dʲ', 'dː', 'd̚', 'd̥', 'd̪', 'd̯', 'd͡z', 'd͡ʑ', 'd͡ʒ', 'd͡ʒː', 'd͡ʒ̤', 'e', 'eː', 'e̞', 'f', 'fʲ', 'fʷ', 'fː', 'g', 'gʲ', 'gʲj', 'gʷ', 'gː', 'h', 'hʷ', 'i', 'ij', 'iː', 'i̞', 'i̥', 'i̯', 'j', 'k', 'kx', 'kʰ', 'kʲ', 'kʲj', 'kʷ', 'kʷʼ', 'kʼ', 'kː', 'k̟ʲ', 'k̟̚', 'k͡p̚', 'l', 'lʲ', 'lː', 'l̪', 'm', 'mʲ', 'mʲj', 'mʷ', 'mː', 'n', 'nj', 'nʲ', 'nː', 'n̪', 'n̺', 'o', 'oː', 'o̞', 'o̥', 'p', 'pf', 'pʰ', 'pʲ', 'pʲj', 'pʷ', 'pʷʼ', 'pʼ', 'pː', 'p̚', 'q', 'r', 'rː', 's', 'sʲ', 'sʼ', 'sː', 's̪', 't', 'ts', 'tsʰ', 'tɕ', 'tɕʰ', 'tʂ', 'tʂʰ', 'tʃ', 'tʰ', 'tʲ', 'tʷʼ', 'tʼ', 'tː', 't̚', 't̪', 't̪ʰ', 't̪̚', 't͡s', 't͡sʼ', 't͡ɕ', 't͡ɬ', 't͡ʃ', 't͡ʃʲ', 't͡ʃʼ', 't͡ʃː', 'u', 'uə', 'uː', 'u͡w', 'v', 'vʲ', 'vʷ', 'vː', 'v̞', 'v̞ʲ', 'w', 'x', 'x̟ʲ', 'y', 'z', 'zj', 'zʲ', 'z̪', 'ä', 'æ', 'ç', 'çj', 'ð', 'ø', 'ŋ', 'ŋ̟', 'ŋ͡m', 'œ', 'œ̃', 'ɐ', 'ɐ̞', 'ɑ', 'ɑ̱', 'ɒ', 'ɓ', 'ɔ', 'ɔ̃', 'ɕ', 'ɕː', 'ɖ̤', 'ɗ', 'ə', 'ɛ', 'ɛ̃', 'ɟ', 'ɡ', 'ɡʲ', 'ɡ̤', 'ɡ̥', 'ɣ', 'ɣj', 'ɤ', 'ɤɐ̞', 'ɤ̆', 'ɥ', 'ɦ', 'ɨ', 'ɪ', 'ɫ', 'ɯ', 'ɯ̟', 'ɯ̥', 'ɰ', 'ɱ', 'ɲ', 'ɳ', 'ɴ', 'ɵ', 'ɸ', 'ɹ', 'ɹ̩', 'ɻ', 'ɻ̩', 'ɽ', 'ɾ', 'ɾj', 'ɾʲ', 'ɾ̠', 'ʀ', 'ʁ', 'ʁ̝', 'ʂ', 'ʃ', 'ʃʲː', 'ʃ͡ɣ', 'ʈ', 'ʉ̞', 'ʊ', 'ʋ', 'ʋʲ', 'ʌ', 'ʎ', 'ʏ', 'ʐ', 'ʑ', 'ʒ', 'ʒ͡ɣ', 'ʔ', 'ʝ', 'ː', 'β', 'β̞', 'θ', 'χ', 'ә', 'ḁ']
    ipa1, ipa2, ipa3 = ipa0.copy(), ipa0.copy(), ipa0.copy()
    random.shuffle(ipa1)
    random.shuffle(ipa2)
    random.shuffle(ipa3)
    # randomly joined to form training data
    passage0 = ' '.join(ipa0)
    passage1 = ' '.join(ipa1)
    passage2 = ' '.join(ipa2)
    passage3 = ' '.join(ipa3)
    data = [passage0, passage1, passage2, passage3]
    # setup
    tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
    # trainer = WordLevelTrainer(vocab_size=300, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    trainer = WordLevelTrainer(vocab_size=300, special_tokens=["<s>","<pad>","</s>","<unk>","<mask>"])
    tokenizer.pre_tokenizer = Whitespace()
    # train the tokenizer
    tokenizer.train_from_iterator(data, trainer=trainer)
    tokenizer.save(args.outdir + '/ipa_tokenizer.json')


if __name__ == "__main__":
    args = get_args()
    main(args)
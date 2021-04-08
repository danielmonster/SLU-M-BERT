import json
import pickle
import subprocess
from allosaurus.app import read_recognizer

model = read_recognizer()
folders = ['map', 'music']
sets = ['test']

def convert(folder, set):
    all_phones = {}

    # get IDs from training set, and find matching recording and translate
    with open(f'../data/{folder}/{set}.json', 'r', encoding='utf-8') as f:
        entries = json.load(f)
        for entry in entries:
            dlg_id = entry['dlg_id']
            for utterence in entry['utterances']:
                wav_id = utterence['wav_id']
                try:
                    transcript = model.recognize(f'../data/{folder}/audios/{wav_id}.wav')
                except Exception:
                    print('Problematic file:', wav_id)
                    continue
                phones_list = [phone for phone in transcript.split()]
                all_phones[wav_id] = phones_list
    
    with open(f'../data/{folder}/{set}_transcript.pkl', 'wb') as ff:
        pickle.dump(all_phones, ff)
    print(f'Successfully transcribed data from ../data/{folder}/{set}.json')

def main():
    for folder in folders:
        for set in sets:
            convert(folder, set)

if __name__ == '__main__':
    main()
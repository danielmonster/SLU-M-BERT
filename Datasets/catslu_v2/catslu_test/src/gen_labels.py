import json
import pickle

def assign_labels(folder, set='test', train_size=0, cut_off=-1):

    with open(f'../data/{folder}/{set}_transcript.pkl', 'rb') as f:
        id_transcript = pickle.load(f, encoding='utf-8')

    train, test = [], []
    with open(f'../data/{folder}/{set}.json', 'r', encoding='utf-8') as f:
        cnt = 0
        entries = json.load(f)
        for entry in entries:
            # break when cut off reached
            if cut_off != -1 and cnt > cut_off:
                break 

            for utterance in entry['utterances']:
                # tracks processed counts
                cnt += 1
                if cut_off != -1 and cnt > cut_off:
                    break 
                wav_id = utterance['wav_id']
                # skips invalid files
                if wav_id not in id_transcript:
                    continue

                transcript = id_transcript[wav_id]
                semantic = utterance['semantic']
                label = get_label(folder, semantic)
                if label != -1:
                    if cnt > train_size:
                        test.append((wav_id, transcript, label))
                    else:
                        train.append((wav_id, transcript, label))
        print(folder, cnt)
    
    # write out training data
    if train_size:
        with open(f'../data/{folder}/labelled_train.csv', 'w', encoding='utf-8') as f:
            f.write('wav_id,transcript,label\n')
            for line in train:
                wav_id, trans_list, label = line
                transcript = ' '.join(trans_list)
                f.write(f'{wav_id},{transcript},{label}\n')
    
    # write out testing data
    with open(f'../data/{folder}/labelled_test.csv', 'w', encoding='utf-8') as f:
        f.write('wav_id,transcript,label\n')
        for line in test:
            wav_id, trans_list, label = line
            transcript = ' '.join(trans_list)
            f.write(f'{wav_id},{transcript},{label}\n')

def get_label(folder, semantic):
    if semantic:
        if folder == 'map':
            return 0
        elif folder == 'music':
            return 1
        elif folder == 'video':
            return 2
        elif folder == 'weather':
            return 3
    # if no semantic, then the input is invalid
    return -1

def main():
    assign_labels('map', cut_off=1000)
    assign_labels('music')
    assign_labels('video', train_size=1000)
    assign_labels('weather', train_size=2000)

if __name__ == '__main__':
    main()
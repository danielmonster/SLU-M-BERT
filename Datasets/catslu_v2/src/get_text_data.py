'''
    The script goes through all train/dev/test.json files
    and extract textual data of the form:
    X - <transcript>
    y - <label>
'''
import json
import re

# --------------- GLOBAL VARIABLES -----------------------
set2path = {
    'train': '../catslu_traindev/data/',
    'development': '../catslu_traindev/data/',
    'test': '../catslu_test/data/'
}

set2list = {
    'train': {
        'man': [],
        'asr': []
    },
    'development': {
        'man': [],
        'asr': []
    },
    'test': {
        'man': [],
        'asr': []
    }
}

# ----------------- HELPER FUNCTIONS ------------------------

def get_data(dataset, category, cut_off=0, split_for_train=0):
    
    # var setups
    target_file = f'{set2path[dataset]}/{category}/{dataset}.json'
    man_list = set2list[dataset]['man']
    asr_list = set2list[dataset]['asr']

    with open(target_file, 'r', encoding='utf-8') as f:
        entries = json.load(f)
        cnt=0 # keep track of utterances read
        rcd=0 # cnt of utterances recorded
        trn=0 # cnt of entries split to train
        for entry in entries:
            if cut_off and cnt > cut_off:
                break # exit if exeeds cutoff

            for utterance in entry['utterances']:
                cnt+=1
                if cut_off and cnt > cut_off:
                    break # exit if exeeds cutoff
                wav_id = utterance['wav_id']

                # check validity
                semantic = utterance['semantic']
                label = get_label(category, semantic)
                if label==-1:
                    continue

                # get transcripts
                man_script = utterance['manual_transcript']
                man_script = re.sub(r'\(\w*\)', '$', man_script)
                asr_script = utterance['asr_1best']

                # append
                if split_for_train and split_for_train>=cnt:
                    set2list['train']['man'].append((wav_id, man_script, label))
                    set2list['train']['asr'].append((wav_id, asr_script, label))
                    trn+=1
                else:
                    man_list.append((wav_id, man_script, label))
                    asr_list.append((wav_id, asr_script, label))
                    rcd+=1
    print(f'{dataset} set of {category} added {rcd} entries')
    print(f'train set of {category} added {trn} entries')

def get_label(category, semantic):
    if semantic:
        if category == 'map' and semantic[0][1] != '页码' and semantic[0][1] != '序列号':
            return 0
        elif category == 'music':
            return 1
        elif category == 'video':
            return 2
        elif category == 'weather':
            return 3
    # if no semantic, then the input is invalid
    return -1

def write_data():
    sets = ['train', 'development', 'test']
    types = ['man', 'asr']
    for type in types:
        for dataset in sets:
            cur_list = set2list[dataset][type]
            dataset = 'dev' if dataset == 'development' else dataset
            with open(f'../preprocessed/{dataset}_{type}.csv', 'w', encoding='utf-8') as f:
                f.write('wav_id,transcript,label\n')
                for item in cur_list:
                    wav_id, transcript, label = item
                    f.write(f'{wav_id},{transcript},{label}\n')

def main():
    categories = ['map', 'music', 'video', 'weather']

    # training data
    get_data('train', 'map', cut_off=4000)
    for cate in categories[1:]:
        get_data('train', cate)

    # dev data
    for cate in categories:
        get_data('development', cate)

    # test data
    get_data('test', 'map')
    get_data('test', 'music')
    get_data('test', 'video', split_for_train=1000)
    get_data('test', 'weather', split_for_train=2000)

    write_data()

if __name__ == '__main__':
    main()
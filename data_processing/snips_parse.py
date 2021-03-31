# Author: Yuexin Li
# Purpose: 
#         (1) Parse (.wav, label_info) from this dataset 
#           (https://github.com/sonos/spoken-language-understanding-research-datasets)
#         (2) Get IPA form of .wav files 


from dataset_handler import CrossValDataset, TrainTestDataset
from future.utils import iteritems
import pickle
import os
import subprocess
import csv

# ---------------------- Global Vars -------------------------
# path
root_dir = "./"
domain = "smart-lights"                         # smart-speaker
lang = "en"                                     # fr
mode = "close"                                  # far
type = "%s-%s-%s" %(domain, lang, mode)
data_dir = "%s%s-%s-%s-field" % (root_dir, domain, lang, mode)
print(data_dir)

# output path
output_path = type + '.csv'

# smart-light label mapping:
CLASSES_light = ["DecreaseBrightness",
            "IncreaseBrightness",
            "SetLightBrightness",
            "SetLightColor", 
            "SwitchLightOff", 
            "SwitchLightOn",]
CLASSES_light_to_LABEL = {CLASSES_light[i]: i for i in range(len(CLASSES_light))}

# smart-speaker label mapping:
CLASSES_speaker =["NextSong",       
            "PreviousSong" ,
            "SpeakerInterrupt" ,
            "ResumeMusic",
            "VolumeDown",
            "VolumeUp",
            "VolumeSet",
            "GetInfos", 
            "PlayMusic"]
CLASSES_speaker_to_LABEL = {CLASSES_speaker[i]: i for i in range(len(CLASSES_speaker))}

# domain mapping
CLASSES = {"smart-lights": CLASSES_light_to_LABEL, "smart-speaker": CLASSES_speaker_to_LABEL}
OBJECT = {"smart-lights": CrossValDataset, "smart-speaker": TrainTestDataset}

# ---------------------- Helper Functions -------------------------
def parse_convert_file(idx, file_path):
    res = []
    file_name = file_path.split("/")[-1]
    
    # get intent, text, label
    info = dataset.get_labels_from_wav(file_name)
    intent, text = info['intent'], info['text']
    label = CLASSES[domain][intent]

    # get phonemic transcription
    command = 'python3 -m allosaurus.run -i ' + file_path
    result = subprocess.check_output(command, shell=True)
    result_list = result.split()
    phones_list = [phone.decode('utf-8') for phone in result_list]
    
    # format res (csv column order)
    res = [idx, phones_list, label, intent, text, type, file_name]
    return res


# output data to csv file
def save_file(rows, path):
    with open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "phones", "label", "intent", "text", "data-type", "file"])
        writer.writerows(rows)


if __name__ == '__main__':
    dataset = OBJECT[domain].from_dir(data_dir)
    rows = []
    i = 0

    for s, wav_file in iteritems(dataset.audio_corpus):
        row = parse_convert_file(i, wav_file)
        rows.append(row)
        if i % 10 == 0:
            print("i=%d" % i)
        i += 1

    
    save_file(rows, output_path)
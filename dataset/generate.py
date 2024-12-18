import csv

from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

sentences_list = []

with open('texts.csv', mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        sentences_list.append(row[0])

i = 1
for sentence in sentences_list:
    print(str(i)+". "+ sentence)
    tts.tts_to_file(text=sentence, file_path="child_inclusion_female/" + str(i) + ".wav", speaker_wav="speakers/child_inclusion_female.mp3", language='it')
    i += 1
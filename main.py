import os
import wave
from pydub import AudioSegment
from pydub.silence import detect_silence
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from speechbrain.dataio.dataio import read_audio
import speechbrain as sb
import json
import numpy as np


def audio_pipeline(wav):
    sig = torchaudio.load(wav)
    return sig


# load sound file for segmenting
sound = AudioSegment.from_file('Sample.wav')
second_segment = sound[290032:463842]

# Detect the silence part of sound file
silences = detect_silence(second_segment, min_silence_len=1000, silence_thresh=-35, seek_step=1)
print(silences)
print(silences[0][1])

# Segmenting the sound segment
temp_names = ["wave1.wav", "wave2.wav", "wave3.wav", "wave4.wav", "wave5.wav", "wave6.wav", "wave7.wav", "wave8.wav",
              "wave9.wav", "wave10.wav"]
sound_segment = sound[:silences[0][0]]
sound_segment.export(temp_names[0], format='wav')
sound_dicts = {
    temp_names[0]: {
        'start': 0,
        'end': silences[0][0]
    }
}
loop_counter = 0
for silence in silences:
    i = loop_counter
    start = silences[i][1]
    if i + 1 >= len(silences):
        break
    end = silences[i + 1][0]
    print('This is ', i + 2, 'th segement: [', start, ',', end, ']')
    loop_counter = loop_counter + 1
    sound_segment = sound[start:end]
    new_dict = {
        temp_names[i + 1]: {
            'start': start,
            'end': end
        }
    }
    sound_dicts.update(**new_dict)
    sound_segment.export(temp_names[i + 1], format='wav')

print(sound_dicts)
with open('segments.json', 'w', encoding='utf-8') as f:
    json.dump(sound_dicts, f, indent=4)

# Loading pre-trained ECAPA-TDNN model
spk_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                           savedir="pretrained_models/spkrec-ecapa-voxceleb")

audio_list = temp_names
output_dir = "features"
# index = torch.arange(192)
for audio in audio_list:
    signal = torchaudio.load(audio)
    signal = signal[0]
    # signal = torch.index_select(signal,1,index=index)
    signal = torch.squeeze(signal)
    features = spk_model.encode_batch(signal)
    features_1d = torch.flatten(features)
    output_path = os.path.join(output_dir, f"{os.path.splitext(audio)[0]}.txt")
    np.savetxt(output_path, features_1d)

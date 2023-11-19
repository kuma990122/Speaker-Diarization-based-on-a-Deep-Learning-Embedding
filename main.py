import glob
import os
from pydub import AudioSegment
from pydub.silence import detect_silence
from pydub.silence import split_on_silence
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import json
import numpy as np
from sklearn.cluster import KMeans

def merge_channels(input_path,output_path):
    sound = AudioSegment.from_file(input_path)
    mono_sound = sound.set_channels(1)
    mono_sound.export(output_path, format='wav')

def remove_record_silence(input_audio_path, mono_audio_path):
    merge_channels(input_audio_path, mono_audio_path)
    audio = AudioSegment.from_file(mono_audio_path)
    segments = split_on_silence(audio, silence_thresh=-40, min_silence_len=1000)
    silences = detect_silence(audio, min_silence_len=1000, silence_thresh=-40, seek_step=1)
    new_audio = AudioSegment.silent(duration=0)
    sound_number = 0
    silence_number = 0
    sound_dict = {}

    output_folder = 'audio/SampleFlac'
    os.makedirs(output_folder, exist_ok=True)

    for segment in segments:
        if sound_number != 0:
            new_audio = segment
            output_file_path = os.path.join(output_folder, f'segment{sound_number}.wav')
            new_audio.export(output_file_path, format='wav')
            print(f'segment{sound_number} is saved')
            sound_number += 1
        else:
            sound_number += 1

    for silence in silences:
        i = silence_number
        start = silences[i][1]
        if i >= len(silences)-1:
            break
        end = silences[i+1][0]
        silence_number = i + 1
        new_dict = {
            f"segment{i+1}": {
                'start': start/1000,
                'end': end/1000
            }
        }
        print(f"segment{i + 1}: start:{start/1000}, end:{end/1000}")
        sound_dict.update(**new_dict)
    sound_number = 0

    with open('segments.json', 'w', encoding='utf-8') as f:
        json.dump(sound_dict, f, indent=4)

# Use glob library to read all the audio files
def get_audio_files(folder_path, allowed_extensions=['.wav', '.mp3', '.flac']):
    audio_files = []
    for ext in allowed_extensions:
        audio_files.extend(glob.glob(os.path.join(folder_path, f'*{ext}')))

    return audio_files

# Segmenting the wav file by removing the silent part, recording information of sound
# Loading the segmented sound file
input_audio_path = 'SampleFlac.flac'
mono_audio_path = 'audio/MonoAudio/monoaudio.wav '
folder_path = 'audio/SampleFlac'
remove_record_silence(input_audio_path,mono_audio_path)
waveform, sample_rate = torchaudio.load('audio/SampleFlac/segment10.wav', normalize=True)

print(f"Sample_rate_one_channel: {sample_rate} Hz")
print(f"Waveform_one_channel: {waveform.shape}")

audio_list = get_audio_files(folder_path)

#Loading the pretrained model
spk_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                           savedir="pretrained_models/spkrec-ecapa-voxceleb")

#Loading the segmented audio list and extracting features and save them as txt
segment_counter = 0
output_dir = "features"
features_list = []

for audio in audio_list:
    segment_counter += 1
    signal = torchaudio.load(audio)
    signal = signal[0]
    # signal = torch.index_select(signal,1,index=index)
    signal = torch.squeeze(signal)
    features = spk_model.encode_batch(signal)
    features_1d = torch.flatten(features)
    features_list.append(features_1d)
    output_path = os.path.join(output_dir, f"Segment_feature{segment_counter}.txt")
    np.savetxt(output_path, features_1d)

all_features_array = np.vstack(features_list)

num_classes = 4
kmeans = KMeans(n_clusters=num_classes,random_state=42)
kmeans.fit(all_features_array)

predicted_labels = kmeans.labels_
print("Predicted Labels: ", predicted_labels)
print(len(predicted_labels))


import glob
import os
import hdbscan
from pydub import AudioSegment
from pydub.silence import detect_silence
from pydub.silence import split_on_silence
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

import Remove_Silent_VAD


def merge_channels(input_path, output_path):
    sound = AudioSegment.from_file(input_path)
    mono_sound = sound.set_channels(1)
    mono_sound.export(output_path, format='wav')


def remove_record_silence(input_audio_path, mono_audio_path):
    # merge_channels(input_audio_path, mono_audio_path)
    audio = AudioSegment.from_file(mono_audio_path)
    segments = split_on_silence(audio, silence_thresh=-53, min_silence_len=750, keep_silence=500)
    silences = detect_silence(audio, min_silence_len=1000, silence_thresh=-53, seek_step=1)
    new_audio = AudioSegment.silent(duration=0)
    sound_number = 0
    silence_number = 0
    sound_dict = {}

    output_folder = 'audio/MonoAudio'
    os.makedirs(output_folder, exist_ok=True)

    for segment in segments:
        new_audio = segment
        output_file_path = os.path.join(output_folder, f'segment{sound_number}.wav')
        new_audio.export(output_file_path, format='wav')
        print(f'segment{sound_number} is saved')
        sound_number += 1

    for silence in silences:
        i = silence_number
        start = silences[i][1]
        if i >= len(silences) - 1:
            break
        end = silences[i + 1][0]
        silence_number = i + 1
        new_dict = {
            f"segment{i + 1}": {
                'start': start / 1000,
                'end': end / 1000
            }
        }
        print(f"segment{i + 1}: start:{start / 1000}, end:{end / 1000}")
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


def feature_extraction_ecapa(audio_list):
    spk_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                               savedir="pretrained_models/spkrec-ecapa-voxceleb")
    segment_counter = 0
    output_dir = "features"
    features_list = []

    for audio in audio_list:
        segment_counter += 1
        waveform, sample_rate = torchaudio.load(audio)
        with torch.no_grad():
            features = spk_model.encode_batch(waveform)
            features_1d = torch.flatten(features)
            features_list.append(features_1d)
            output_path = os.path.join(output_dir, f"Segment_feature{segment_counter}.txt")
            np.savetxt(output_path, features_1d)
    return features_list


def feature_extraction_xvec(audio_list):
    spk_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
                                               savedir="pretrained_models/spkrec-xvect-voxceleb")
    # Loading the segmented audio list and extracting features and save them as txt
    segment_counter = 0
    output_dir = "features"
    features_list = []

    for audio in audio_list:
        segment_counter += 1
        waveform, sample_rate = torchaudio.load(audio)
        with torch.no_grad():
            features = spk_model.encode_batch(waveform)
            features_1d = torch.flatten(features)
            features_list.append(features_1d)
            output_path = os.path.join(output_dir, f"Segment_feature{segment_counter}.txt")
            np.savetxt(output_path, features_1d)
    return features_list


def clustering_K_Means(all_features_array, num_classes, model_name):
    kmeans = KMeans(n_clusters=num_classes, random_state=42)
    kmeans.fit(all_features_array)
    kmeans_labels = kmeans.predict(all_features_array)
    # predicted_labels = kmeans.labels_
    print("K-Means Labels with ", model_name, " ", kmeans_labels)
    print("# of labels: ", len(kmeans_labels))

    return kmeans_labels


def clustering_HdbScan(all_features_array, num_cluster, model_name):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=num_cluster, gen_min_span_tree=True)
    cluster_labels = clusterer.fit_predict(all_features_array)
    print(f"Hdbscan Labels with ", model_name, " ", cluster_labels)
    print("# of labels: ", len(cluster_labels))
    return cluster_labels


def visualize_clustering(features, labels_list, num_samples, clusterAlgo_model):
    unique_labels = np.unique(labels_list)
    model = RandomForestClassifier(random_state=42)
    model.fit(features, labels_list)
    feature_importances = model.feature_importances_
    top_two_features_indices = np.argsort(feature_importances)[-2:]
    plt.figure(figsize=(15, 8))

    for label in unique_labels:
        label_indices = np.where(labels_list == label)[0]

        if len(label_indices) <= num_samples:
            selected_indices = label_indices
        else:
            selected_indices = np.random.choice(label_indices, size=num_samples, replace=False)

        selected_features = features[selected_indices]

        selected_features_reshaped = selected_features.reshape(-1, features.shape[1])

        plt.scatter(selected_features_reshaped[:, top_two_features_indices[0]],
                    selected_features_reshaped[:, top_two_features_indices[1]], label=f'Cluster {label}')

    plt.title(f"{clusterAlgo_model[0]} Clustering Visualization with {clusterAlgo_model[1]}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()


ECAPA_TDNN = 'ECAPA_TDNN'
X_VECTOR = 'X_VECTOR'
input_audio_path = 'SampleVAD.wav'
mono_audio_path = 'sound/SampleVAD.wav'
wave_path = "sound"
Remove_Silent_VAD.remove_silent_VAD(wave_path)
# Loading the segmented sound file
folder_path = 'audio/MonoAudio'
audio_list = get_audio_files(folder_path)
audio_length = len(audio_list)

# Running ECAPA for audio list, then use K-Means&HDBSCAN to cluster
array_feature_ecapa = feature_extraction_ecapa(audio_list)
all_features_array_ecapa = np.vstack(array_feature_ecapa)
KMeans_labels_ecapa = clustering_K_Means(all_features_array_ecapa, 2, ECAPA_TDNN)
Hdb_labels_ecapa = clustering_HdbScan(all_features_array_ecapa, 2, ECAPA_TDNN)

# print out the result of clustering, visualize both result
print("ECAPA Features dimensions: ", np.ndim(all_features_array_ecapa))
print("ECAPA Labels dimensions: ", np.ndim(KMeans_labels_ecapa))
print("ECAPA Features shape: ", np.shape(all_features_array_ecapa))
visualize_clustering(all_features_array_ecapa, KMeans_labels_ecapa, audio_length, ["K-Means", "ECAPA-TDNN"])
visualize_clustering(all_features_array_ecapa, Hdb_labels_ecapa, audio_length, ["HDBSCAN", "ECAPA-TDNN"])

# Evaluate the different clustering result with ECAPA-TDNN
silhouette_avg_K_Means_ecapa = silhouette_score(all_features_array_ecapa, KMeans_labels_ecapa)
silhouette_avg_HDBSCAN_ecapa = silhouette_score(all_features_array_ecapa, Hdb_labels_ecapa)
print(f"Average Silhouette Score of K-Means with {ECAPA_TDNN}: {silhouette_avg_K_Means_ecapa}")
print(f"Average Silhouette Score of HDBSCAN with {ECAPA_TDNN}: {silhouette_avg_HDBSCAN_ecapa}")

# Running X-Vector for audio list, then use K-Means&HDBSCAN to cluster
array_feature_xvec = feature_extraction_xvec(audio_list)
all_features_array_xvec = np.vstack(array_feature_xvec)
KMeans_labels_xvec = clustering_K_Means(all_features_array_xvec, 2, X_VECTOR)
HDB_labels_xvec = clustering_HdbScan(all_features_array_xvec, 2, X_VECTOR)

# print out the result of clustering, visualize both result
print("X-Vector Features dimensions: ", np.ndim(all_features_array_xvec))
print("X-Vector Labels dimensions: ", np.ndim(KMeans_labels_xvec))
print("X-Vector Features shape: ", np.shape(all_features_array_xvec))
visualize_clustering(all_features_array_xvec, KMeans_labels_xvec, audio_length, ["K-Means", "X-Vector"])
visualize_clustering(all_features_array_xvec, HDB_labels_xvec, audio_length, ["HDBSCAN", "X-Vector"])

# Evaluate the different clustering result with X-VECTOR
silhouette_avg_K_Means_xvec = silhouette_score(all_features_array_xvec, KMeans_labels_xvec)
silhouette_avg_HDBSCAN_xvec = silhouette_score(all_features_array_xvec, HDB_labels_xvec)
print(f"Average Silhouette Score of K-Means with X-VECTOR: {silhouette_avg_K_Means_xvec}")
print(f"Average Silhouette Score of HDBSCAN with X-VECTOR: {silhouette_avg_HDBSCAN_xvec}")

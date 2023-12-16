import wave
import os
from speechbrain.pretrained import VAD


def remove_silent_VAD(path):
    wav_path = path
    # Open the WAV file
    vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty")
    for k, (dirpath, dirnames, filenames) in enumerate(os.walk(wav_path)):
        print("i value", k)
        l = len(filenames)
    for f in range(l):
        print(f)
        path = dirpath + "/" + filenames[f]
        # rate, audio = read(path)
        audio_file = path
        prob_chunks = vad.get_speech_prob_file(audio_file)
        # 2- Let's apply a threshold on top of the posteriors
        prob_th = vad.apply_threshold(prob_chunks,activation_th=0.500).float()
        # 3- Let's now derive the candidate speech segments
        boundaries1 = vad.get_boundaries(prob_th)
        # 4- Apply energy VAD within each candidate speech segment (optional)
        boundaries2 = vad.energy_VAD(audio_file, boundaries1,activation_th=0.500)
        # 5- Merge segments that are too close
        boundaries3 = vad.merge_close_segments(boundaries2, close_th=0.450)
        # 6- Remove segments that are too short
        boundaries4 = vad.remove_short_segments(boundaries3, len_th=0.400)
        # 7- Double-check speech segments (optiox`nal).
        boundaries5 = vad.double_check_speech_segments(boundaries4, audio_file, speech_th=0.500)
        print(boundaries5)
        array = boundaries5.numpy()
        L = len(array)
        with wave.open(audio_file, 'rb') as wav_file:
            # Get the sample rate and number of channels
            sample_rate = wav_file.getframerate()
            num_channels = wav_file.getnchannels()
            wav_file_silent_remove = bytearray()
            for i in range(L):
                # Set the start and end points (in seconds)
                start_sec = array[i, 0]
                end_sec = array[i, 1]
                # Convert the start and end points to frames
                start_frame = int(start_sec * sample_rate)
                end_frame = int(end_sec * sample_rate)
                # Set the frame range to extract
                wav_file.setpos(start_frame)
                frame_range = end_frame - start_frame
                # Read the frames from the WAV file
                frames = wav_file.readframes(frame_range)
                # wav_file_silent_remove+=frames
                # Write the extracted frames to a new WAV file
                str_pointer = str(i)
                # ide=str(f)
                path = "E:/Studying" + "/" + "MSC program"+"/"+"Project Laboratory 1"+"/"+"Project"+"/"+"audio"+"/"+"MonoAudio"+"/"+str_pointer + "_" + filenames[f]
                with wave.open(path, 'wb') as extracted_file:
                    extracted_file.setnchannels(num_channels)
                    extracted_file.setframerate(sample_rate)
                    extracted_file.setsampwidth(wav_file.getsampwidth())
                    extracted_file.writeframes(frames)

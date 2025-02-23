import numpy as np
import librosa
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from pydub import AudioSegment
from scipy.signal import butter, lfilter
import subprocess
import os  # 추가

# 학습된 모델 로드
best_model = load_model("best_model.h5")

# LabelEncoder 로드 (학습 시 사용한 것과 동일한 인코더 사용)
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_classes.npy', allow_pickle=True)

# 고역 필터 생성 및 적용 함수 정의
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def remove_noise(audio_data, sample_rate):
    cutoff_frequency = 1000  # Hz, 필요에 따라 조정 가능
    filtered_data = highpass_filter(audio_data, cutoff_frequency, sample_rate)
    return filtered_data

def convert_m4a_to_wav(input_file, output_file):
    """M4A 파일을 WAV 파일로 변환합니다."""
    command = f"ffmpeg -i {input_file} -vn -acodec pcm_s16le -ar 44100 -ac 2 {output_file}"
    subprocess.call(command, shell=True)

def extract_features(file_path):
    """오디오 파일에서 특징을 추출합니다."""
    try:
        audio_segment = AudioSegment.from_file(file_path, format="wav")
        audio_samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
        audio_samples /= (2**15)  # Normalize to range [-1, 1]

        # 노이즈 제거
        cleaned_samples = remove_noise(audio_samples, audio_segment.frame_rate)

        # MFCC 특징 추출
        mfccs = librosa.feature.mfcc(y=cleaned_samples, sr=audio_segment.frame_rate, n_mfcc=40)
        spectral_centroid = librosa.feature.spectral_centroid(y=cleaned_samples, sr=audio_segment.frame_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=cleaned_samples, sr=audio_segment.frame_rate)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=cleaned_samples)

        zero_crossing_rate_reshaped = zero_crossing_rate.reshape(1, -1)
        spectral_centroid_reshaped = spectral_centroid.reshape(1, -1)
        spectral_rolloff_reshaped = spectral_rolloff.reshape(1, -1)

        features = np.concatenate((mfccs, spectral_centroid_reshaped, spectral_rolloff_reshaped, zero_crossing_rate_reshaped), axis=0)

        max_pad_len = 174
        pad_width = max_pad_len - features.shape[1]
        if pad_width >= 0:
            features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            st.warning(f"Warning: File {file_path} is too long. Truncating to {max_pad_len} frames.")
            features = features[:, :max_pad_len]

        return features

    except Exception as e:
        st.error(f"Failed to process audio: {file_path}, error: {str(e)}")
        return None

def predict_audio(file_path):
    """오디오 파일을 예측합니다."""
    features = extract_features(file_path)
    if features is None:
        raise ValueError("파일의 특징 추출에 실패했습니다.")

    features = np.expand_dims(features, axis=0)  # 배치 차원 추가
    features = np.expand_dims(features, axis=-1)  # 채널 차원 추가

    predictions = best_model.predict(features)
    predicted_index = np.argmax(predictions)
    predicted_label = label_encoder.inverse_transform([predicted_index])

    return predicted_label[0]

# Streamlit 앱
st.title('Audio Classification App')

uploaded_file = st.file_uploader("Upload an audio file (M4A or WAV)", type=["m4a", "wav"], key="audio_uploader")

if uploaded_file is not None:
    file_path = "temp.wav"
    if uploaded_file.name.endswith('.m4a'):
        # M4A 파일을 WAV 파일로 변환
        with open("temp.m4a", "wb") as f:
            f.write(uploaded_file.getbuffer())
        convert_m4a_to_wav("temp.m4a", file_path)
        os.remove("temp.m4a")  # 변환 후 불필요한 M4A 파일 삭제
    else:
        # WAV 파일 직접 저장
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.audio(file_path, format='audio/wav')
    try:
        predicted_class = predict_audio(file_path)
        st.success(f"Predicted class for uploaded file: {predicted_class}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        os.remove(file_path)  # 임시 파일 삭제
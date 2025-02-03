import subprocess
import sys

def install_requirements():
    requirements = [
        'pandas',
        'numpy',
        'tensorflow',
        'nltk',
        'scikit-learn',
        'summa'
    ]
    
    print("필요한 라이브러리 설치 중...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except:
            print(f"{package} 설치 중 오류가 발생했습니다.")
            sys.exit(1)
    print("라이브러리 설치 완료\n")

# 필요한 라이브러리 설치
install_requirements()

# 필요한 라이브러리 import
import urllib.request
import pandas as pd
import numpy as np
from importlib.metadata import version
import nltk
import tensorflow as tf
from summa import summarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore # type: ignore
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Attention # type: ignore # type: ignore
from tensorflow.keras.models import Model # type: ignore
import re

# NLTK 데이터 다운로드
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except:
    print("NLTK 데이터 다운로드 중 오류가 발생했습니다.")

# GPU 메모리 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 1. 데이터 수집
print("라이브러리 버전 확인:")
print(f"NLTK: {nltk.__version__}")
print(f"TensorFlow: {tf.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Summa: {version('summa')}")

# 데이터 다운로드
try:
    print("데이터 다운로드 중...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/sunnysai12345/News_Summary/master/news_summary_more.csv",
        filename="news_summary_more.csv"
    )
    data = pd.read_csv('news_summary_more.csv', encoding='iso-8859-1')
    print("데이터 다운로드 완료")
except Exception as e:
    print(f"데이터 다운로드 중 오류 발생: {e}")
    sys.exit(1)

# 2. 데이터 전처리 함수
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # 소문자 변환
    text = text.lower()
    # 특수문자 제거
    text = re.sub(r'[^\w\s]', '', text)
    # 숫자 제거
    text = re.sub(r'\d+', '', text)
    # 여러 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 데이터 크기 제한 (메모리 관리를 위해)
MAX_SAMPLES = 1000
data = data.head(MAX_SAMPLES)

print("데이터 전처리 시작...")
# 데이터 전처리 적용
data['cleaned_text'] = data['text'].apply(preprocess_text)
data['cleaned_headlines'] = data['headlines'].apply(preprocess_text)
print("데이터 전처리 완료")

# 3. 시퀀스 준비
MAX_TEXT_LEN = 200
MAX_SUMMARY_LEN = 30
VOCAB_SIZE = 20000

# 토크나이저 생성
print("토크나이저 학습 시작...")
text_tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
summary_tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")

# 텍스트와 요약문에 대해 토크나이저 학습
text_tokenizer.fit_on_texts(data['cleaned_text'])
summary_tokenizer.fit_on_texts(data['cleaned_headlines'])

# 시퀀스 변환
X = text_tokenizer.texts_to_sequences(data['cleaned_text'])
y = summary_tokenizer.texts_to_sequences(data['cleaned_headlines'])

# 패딩
X = pad_sequences(X, maxlen=MAX_TEXT_LEN, padding='post')
y = pad_sequences(y, maxlen=MAX_SUMMARY_LEN, padding='post')
print("토크나이저 학습 완료")

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 어텐션 기반 Seq2Seq 모델 생성
def create_model(vocab_size, max_text_len, max_summary_len):
    # 인코더
    encoder_inputs = Input(shape=(max_text_len,))
    enc_emb = Embedding(vocab_size, 256)(encoder_inputs)
    encoder_lstm = LSTM(256, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
    
    # 디코더
    decoder_inputs = Input(shape=(max_summary_len,))
    dec_emb = Embedding(vocab_size, 256)(decoder_inputs)
    decoder_lstm = LSTM(256, return_sequences=True)
    decoder_outputs = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
    
    # 어텐션 레이어
    attention = Attention()([decoder_outputs, encoder_outputs])
    
    # 출력 레이어
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(attention)
    
    # 모델 생성
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 모델 생성 및 학습
print("모델 생성 중...")
model = create_model(VOCAB_SIZE, MAX_TEXT_LEN, MAX_SUMMARY_LEN)
print("모델 학습 시작...")
try:
    # 배치 크기 줄이기
    BATCH_SIZE = 16
    
    # 메모리 효율적인 데이터 생성
    decoder_input_data = np.zeros((len(y_train), MAX_SUMMARY_LEN))
    
    # 디코더 입력 준비
    for i, seq in enumerate(y_train):
        decoder_input_data[i] = np.pad(seq, (0, MAX_SUMMARY_LEN - len(seq)), 'constant')
    
    # 모델 학습
    model.fit(
        [X_train, decoder_input_data], 
        y_train,
        epochs=10,
        batch_size=BATCH_SIZE,
        validation_split=0.2
    )
    print("모델 학습 완료")
except Exception as e:
    print(f"모델 학습 중 오류 발생: {e}")
    sys.exit(1)

# 5. 추상적 요약 생성 및 평가
def generate_summary(text):
    try:
        # 텍스트 전처리
        cleaned_text = preprocess_text(text)
        
        # 시퀀스 변환
        text_seq = text_tokenizer.texts_to_sequences([cleaned_text])
        padded_text = pad_sequences(text_seq, maxlen=MAX_TEXT_LEN, padding='post')
        
        # 예측
        predicted = model.predict([padded_text, np.zeros((1, MAX_SUMMARY_LEN))])
        predicted_text = ""
        for word_idx in predicted[0]:
            word = summary_tokenizer.index_word.get(np.argmax(word_idx), '')
            if word != '':
                predicted_text += word + ' '
        
        return predicted_text.strip()
    except Exception as e:
        print(f"요약 생성 중 오류 발생: {e}")
        return ""

# 6. Summa를 이용한 추출적 요약
def extractive_summary(text, ratio=0.3):
    try:
        return summarizer.summarize(text, ratio=ratio)
    except Exception as e:
        print(f"추출적 요약 중 오류 발생: {e}")
        return ""

# 테스트
print("\n=== 요약 테스트 ===")
try:
    # 테스트 세트의 일부만 사용
    test_samples = min(100, len(X_test))
    generated_summaries = []
    test_texts = []
    test_headlines = []
    
    print(f"\n{test_samples}개의 테스트 샘플에 대해 요약 생성 중...")
    
    # 테스트 데이터 준비
    test_indices = np.random.choice(len(X_test), test_samples, replace=False)
    
    for i, idx in enumerate(test_indices):
        # 원본 텍스트와 헤드라인 가져오기
        original_idx = np.where((X == X_test[idx]).all(axis=1))[0][0]
        test_text = data['text'].iloc[original_idx]
        test_headline = data['headlines'].iloc[original_idx]
        
        # 리스트에 추가
        test_texts.append(test_text)
        test_headlines.append(test_headline)
        
        # 요약문 생성
        generated_summary = generate_summary(test_text)
        generated_summaries.append(generated_summary)
        
        # 진행상황 출력
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{test_samples} 완료")
    
    # 테스트 데이터 저장
    with open('test_data.txt', 'w', encoding='utf-8') as f:
        for text, headline in zip(test_texts, test_headlines):
            f.write(f"{text}\t{headline}\n")
    
    # 생성된 요약문 저장
    with open('generated_summaries.txt', 'w', encoding='utf-8') as f:
        for summary in generated_summaries:
            f.write(summary + '\n')
    
    # 샘플 출력
    print("\n=== 샘플 요약 ===")
    for i in range(min(3, len(generated_summaries))):
        print(f"\n=== 샘플 {i+1} ===")
        print("원본 텍스트:", test_texts[i][:200], "...")
        print("\n원본 요약:", test_headlines[i])
        print("\n생성된 요약:", generated_summaries[i])
        print("\n추출적 요약:", extractive_summary(test_texts[i]))

except Exception as e:
    print(f"테스트 중 오류 발생: {e}") 
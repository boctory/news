import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow 경고 숨기기

import pandas as pd # type: ignore
import numpy as np # type: ignore
import nltk # type: ignore
import tensorflow as tf # type: ignore
from summa import summarizer # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense # type: ignore
from tensorflow.keras.models import Model # type: ignore
import re
import sys
from importlib.metadata import version
import urllib.request
import nltk.data # type: ignore

# NLTK 다운로드 메시지 숨기기
nltk.data.path.append(os.path.join(os.path.expanduser("~"), "nltk_data"))
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

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
MAX_TEXT_LEN = 50
MAX_SUMMARY_LEN = 10
VOCAB_SIZE = 5000

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

# 간단한 Seq2Seq 모델로 변경
def create_model(vocab_size, max_text_len, max_summary_len):
    # 인코더
    encoder_inputs = Input(shape=(max_text_len,))
    enc_emb = Embedding(vocab_size, 64)(encoder_inputs)  # 임베딩 차원 축소
    encoder = LSTM(64, return_state=True)  # LSTM 단순화
    encoder_outputs, state_h, state_c = encoder(enc_emb)
    
    # 디코더
    decoder_inputs = Input(shape=(max_summary_len,))
    dec_emb = Embedding(vocab_size, 64)(decoder_inputs)
    decoder_lstm = LSTM(64, return_sequences=True)
    decoder_outputs = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
    
    # 출력 레이어
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # 모델 생성
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    # 컴파일
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # 학습률 조정
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 모델 생성 및 학습
print("모델 생성 중...")
model = create_model(VOCAB_SIZE, MAX_TEXT_LEN, MAX_SUMMARY_LEN)
print("모델 학습 시작...")
try:
    BATCH_SIZE = 64  # 배치 크기 증가
    EPOCHS = 3  # 에폭 수 감소
    
    # 디코더 입력 데이터 준비
    decoder_input_data = np.zeros_like(y_train)
    decoder_input_data[:, 1:] = y_train[:, :-1]
    
    # 클리핑을 적용한 옵티마이저 설정
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0  # 그래디언트 클리핑 추가
    )
    
    # 모델 재컴파일
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 모델 학습
    history = model.fit(
        [X_train, decoder_input_data],
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1
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
        
        # 디코더 입력 준비 (시작 토큰으로 시작)
        decoder_input = np.zeros((1, MAX_SUMMARY_LEN))
        
        # 요약문 생성
        generated_text = []
        for i in range(MAX_SUMMARY_LEN):
            # 예측
            output = model.predict([padded_text, decoder_input], verbose=0)
            # 다음 단어 선택
            predicted_word_idx = np.argmax(output[0, i])
            # 예측된 단어를 리스트에 추가
            if predicted_word_idx > 0:  # 0은 패딩
                word = summary_tokenizer.index_word.get(predicted_word_idx, '')
                if word:
                    generated_text.append(word)
        
        # 생성된 단어들을 문장으로 조합
        return ' '.join(generated_text)
    except Exception as e:
        print(f"요약 생성 중 오류 발생: {e}")
        return "요약 생성 실패"

# 6. Summa를 이용한 추출적 요약
def extractive_summary(text, ratio=0.3):
    try:
        summary = summarizer.summarize(text, ratio=ratio)
        if not summary:  # 요약이 비어있으면
            return text[:200] + "..."  # 첫 200자를 반환
        return summary
    except Exception as e:
        print(f"추출적 요약 중 오류 발생: {e}")
        return text[:200] + "..."  # 오류 발생시 첫 200자를 반환

def test_model():
    with open('output.txt', 'w', encoding='utf-8') as f:
        f.write("=== 요약 생성 결과 ===\n")
        
        # 테스트할 샘플 수 제한
        num_samples = min(3, len(X_test))
        
        for i in range(num_samples):
            # 원본 텍스트와 요약 가져오기
            original_idx = np.where((X == X_test[i]).all(axis=1))[0][0]
            test_text = data['text'].iloc[original_idx]
            test_headline = data['headlines'].iloc[original_idx]
            
            # 요약 생성
            print(f"\n[샘플 {i+1} 요약 생성 중...]")
            generated_summary = generate_summary(test_text)
            extractive_result = extractive_summary(test_text)
            
            # 결과 저장 및 출력
            output_text = (
                f"\n[샘플 {i+1}]\n"
                f"원문 (앞부분): {test_text[:200]}...\n"
                f"원본 요약: {test_headline}\n"
                f"생성된 요약: {generated_summary}\n"
                f"추출적 요약: {extractive_result}\n"
            )
            
            # 파일에 저장
            f.write(output_text)
            # 콘솔에 출력
            print(output_text)

# 그래프 관련 코드 제거 

def main():
    # 모델 테스트 실행
    test_model()

    # 추출적 요약 테스트
    print("\n=== 추출적 요약 결과 ===")
    test_samples = min(3, len(data))
    
    for i in range(test_samples):
        text = data['text'].iloc[i]
        original_summary = data['headlines'].iloc[i]
        summary = extractive_summary(text)
        
        output_text = (
            f"\n[샘플 {i+1}]\n"
            f"원문 (앞부분): {text[:200]}...\n"
            f"원본 요약: {original_summary}\n"
            f"생성된 요약: {summary}\n"
            f"---\n"
        )
        
        # 결과를 파일에 저장
        with open('output.txt', 'a', encoding='utf-8') as f:
            f.write(output_text)
        
        # 콘솔에 출력
        print(output_text)

if __name__ == "__main__":
    main() 
import subprocess
import sys

def install_requirements():
    requirements = [
        'pandas',
        'numpy',
        'tensorflow',
        'rouge-score',
        'matplotlib',
        'seaborn'
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

import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model # type: ignore
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore

def load_data(filename='news_summary_more.csv'):
    try:
        data = pd.read_csv(filename, encoding='iso-8859-1')
        print(f"데이터 로드 완료: {len(data)}개의 샘플")
        return data
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        sys.exit(1)

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def evaluate_summaries(original_summaries, generated_summaries):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {
        'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
        'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
        'rougeL': {'precision': [], 'recall': [], 'fmeasure': []}
    }
    
    for orig, gen in zip(original_summaries, generated_summaries):
        score = scorer.score(orig, gen)
        for metric in scores.keys():
            scores[metric]['precision'].append(score[metric].precision)
            scores[metric]['recall'].append(score[metric].recall)
            scores[metric]['fmeasure'].append(score[metric].fmeasure)
    
    return scores

def plot_scores(scores):
    plt.figure(figsize=(15, 5))
    
    metrics = ['precision', 'recall', 'fmeasure']
    rouge_types = list(scores.keys())
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        data = [scores[rouge_type][metric] for rouge_type in rouge_types]
        plt.boxplot(data, labels=rouge_types)
        plt.title(f'Distribution of {metric}')
        plt.ylabel('Score')
    
    plt.tight_layout()
    plt.savefig('rouge_scores.png')
    print("ROUGE 점수 분포도가 'rouge_scores.png'로 저장되었습니다.")

def print_detailed_comparison(original, generated, scores, index):
    print(f"\n=== 샘플 {index + 1} 상세 비교 ===")
    print(f"원본 요약: {original}")
    print(f"생성된 요약: {generated}")
    print("\nROUGE 점수:")
    for metric, values in scores.items():
        print(f"\n{metric}:")
        for key, value in values.items():
            print(f"  {key}: {value[index]:.4f}")

def load_test_data(filename='test_data.txt'):
    try:
        texts = []
        headlines = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # 빈 줄 건너뛰기
                    continue
                    
                # 탭으로 구분된 첫 번째 구분자만 사용
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    text, headline = parts
                    if text and headline:  # 둘 다 비어있지 않은 경우만 추가
                        texts.append(text)
                        headlines.append(headline)
                else:
                    print(f"Warning: 잘못된 형식의 라인 발견: {line[:100]}...")
        
        if not texts:
            raise ValueError("데이터를 읽을 수 없습니다.")
            
        print(f"테스트 데이터 로드 완료: {len(texts)}개의 샘플")
        return texts, headlines
    except Exception as e:
        print(f"테스트 데이터 로드 중 오류 발생: {e}")
        sys.exit(1)

def load_generated_summaries(filename='generated_summaries.txt'):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            summaries = [line.strip() for line in f if line.strip()]
        print(f"생성된 요약문 로드 완료: {len(summaries)}개의 요약")
        return summaries
    except Exception as e:
        print(f"생성된 요약문 로드 중 오류 발생: {e}")
        sys.exit(1)

def main():
    # 테스트 데이터 로드
    test_texts, original_summaries = load_test_data()
    
    # 생성된 요약문 로드
    generated_summaries = load_generated_summaries()
    
    # 데이터 개수 확인
    if len(original_summaries) != len(generated_summaries):
        print(f"Warning: 원본 요약문({len(original_summaries)}개)과 생성된 요약문({len(generated_summaries)}개)의 개수가 일치하지 않습니다.")
        # 더 작은 수로 맞추기
        min_len = min(len(original_summaries), len(generated_summaries))
        original_summaries = original_summaries[:min_len]
        generated_summaries = generated_summaries[:min_len]
        print(f"평가를 위해 {min_len}개의 샘플만 사용합니다.")
    
    # ROUGE 점수 계산
    print("\n=== ROUGE 점수 계산 중 ===")
    scores = evaluate_summaries(original_summaries, generated_summaries)
    
    # 결과 출력
    print("\n=== 전체 ROUGE 점수 통계 ===")
    for metric in scores.keys():
        print(f"\n{metric}:")
        for key in scores[metric].keys():
            values = scores[metric][key]
            print(f"  {key}:")
            print(f"    평균: {np.mean(values):.4f}")
            print(f"    중앙값: {np.median(values):.4f}")
            print(f"    표준편차: {np.std(values):.4f}")
    
    # 상세 비교 (처음 3개 샘플)
    for i in range(min(3, len(original_summaries))):
        print_detailed_comparison(
            original_summaries[i],
            generated_summaries[i],
            scores,
            i
        )
        print("\n원본 텍스트:", test_texts[i][:200], "...")
    
    # 시각화
    plot_scores(scores)

if __name__ == "__main__":
    main() 
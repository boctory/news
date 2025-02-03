import subprocess
import sys

def install_requirements():
    requirements = [
        'pandas',
        'summa',
        'nltk',
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
from summa import summarizer
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import seaborn as sns
import re

def load_data(filename='news_summary_more.csv'):
    try:
        data = pd.read_csv(filename, encoding='iso-8859-1')
        print(f"데이터 로드 완료: {len(data)}개의 샘플")
        return data
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        sys.exit(1)

def extractive_summary(text, ratio=0.3):
    try:
        return summarizer.summarize(text, ratio=ratio)
    except Exception as e:
        print(f"추출적 요약 중 오류 발생: {e}")
        return ""

def evaluate_summaries(original_summaries, generated_summaries):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {
        'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
        'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
        'rougeL': {'precision': [], 'recall': [], 'fmeasure': []}
    }
    
    for orig, gen in zip(original_summaries, generated_summaries):
        if not gen:  # 빈 요약문 처리
            for metric in scores.keys():
                scores[metric]['precision'].append(0)
                scores[metric]['recall'].append(0)
                scores[metric]['fmeasure'].append(0)
            continue
            
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
    plt.savefig('extractive_rouge_scores.png')
    print("ROUGE 점수 분포도가 'extractive_rouge_scores.png'로 저장되었습니다.")

def clean_text_for_save(text):
    if not isinstance(text, str):
        return ""
    # 개행 문자와 탭 문자를 공백으로 대체
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # 연속된 공백을 하나로
    text = ' '.join(text.split())
    return text

def main():
    # 데이터 로드
    data = load_data()
    
    # 테스트 샘플 수 설정
    test_samples = min(100, len(data))
    original_summaries = []
    generated_summaries = []
    
    print(f"\n{test_samples}개의 테스트 샘플에 대해 추출적 요약 생성 중...")
    
    # 다양한 ratio 값으로 실험
    ratios = [0.1, 0.2, 0.3]
    best_ratio = 0.3
    best_avg_score = 0
    
    # 테스트할 인덱스 미리 선택
    test_indices = list(range(test_samples))
    
    # 원본 요약문 미리 수집
    original_summaries = [data['headlines'].iloc[i] for i in test_indices]
    
    for ratio in ratios:
        print(f"\nratio = {ratio}로 테스트 중...")
        current_summaries = []
        
        for i in test_indices:
            text = data['text'].iloc[i]
            generated_summary = extractive_summary(text, ratio)
            current_summaries.append(generated_summary)
            
            if i < 3:  # 처음 3개 샘플만 출력
                print(f"\n=== 샘플 {i+1} ===")
                print("원본 텍스트:", text[:200], "...")
                print("\n원본 요약:", original_summaries[i])
                print("\n생성된 요약:", generated_summary)
        
        # ROUGE 점수 계산
        scores = evaluate_summaries(original_summaries, current_summaries)
        avg_score = np.mean([np.mean(scores[metric]['fmeasure']) for metric in scores.keys()])
        
        print(f"\n=== ratio {ratio}의 ROUGE 점수 ===")
        for metric in scores.keys():
            print(f"\n{metric}:")
            for key in scores[metric].keys():
                values = scores[metric][key]
                print(f"  {key}:")
                print(f"    평균: {np.mean(values):.4f}")
                print(f"    중앙값: {np.median(values):.4f}")
                print(f"    표준편차: {np.std(values):.4f}")
        
        # 최적의 ratio 선택
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            best_ratio = ratio
            generated_summaries = current_summaries
    
    print(f"\n최적의 ratio: {best_ratio} (평균 ROUGE F1: {best_avg_score:.4f})")
    
    # 최종 결과 저장
    print("\n결과 저장 중...")
    
    # 생성된 요약문 저장
    with open('generated_summaries.txt', 'w', encoding='utf-8') as f:
        for summary in generated_summaries:
            cleaned_summary = clean_text_for_save(summary)
            f.write(f"{cleaned_summary}\n")
    
    # 테스트 데이터 저장
    with open('test_data.txt', 'w', encoding='utf-8') as f:
        for i in test_indices:
            text = clean_text_for_save(data['text'].iloc[i])
            headline = clean_text_for_save(data['headlines'].iloc[i])
            if text and headline:  # 빈 문자열이 아닌 경우만 저장
                f.write(f"{text}\t{headline}\n")
    
    print("결과 저장 완료")
    
    # 최종 시각화
    final_scores = evaluate_summaries(original_summaries, generated_summaries)
    plot_scores(final_scores)

if __name__ == "__main__":
    main() 
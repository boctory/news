import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow 경고 숨기기

import pandas as pd # type: ignore
import numpy as np # type: ignore
from summa import summarizer # type: ignore
from rouge_score import rouge_scorer # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import re
import sys

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
        # 텍스트 전처리
        text = text.replace('\n', ' ').strip()
        
        # 요약 생성
        summary = summarizer.summarize(text, ratio=ratio)
        
        # 요약이 비어있으면 다른 비율로 시도
        if not summary:
            for r in [0.4, 0.5, 0.6]:
                summary = summarizer.summarize(text, ratio=r)
                if summary:
                    break
        
        # 여전히 비어있으면 원문의 첫 부분 반환
        if not summary:
            sentences = text.split('.')
            summary = '. '.join(sentences[:3]) + '.'
            
        return summary
    except Exception as e:
        print(f"추출적 요약 중 오류 발생: {e}")
        # 오류 발생시 원문의 첫 3문장 반환
        sentences = text.split('.')
        return '. '.join(sentences[:3]) + '.'

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
            f"추출적 요약: {summary}\n"
            f"---\n"
        )
        
        # 파일에 저장
        with open('output.txt', 'a', encoding='utf-8') as f:
            f.write(output_text)
        
        # 콘솔에 출력
        print(output_text)

if __name__ == "__main__":
    main() 
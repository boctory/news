import pandas as pd # type: ignore
import numpy as np # type: ignore
from rouge_score import rouge_scorer # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from summa import summarizer # type: ignore
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
        score = scorer.score(orig, gen)
        for metric in scores.keys():
            scores[metric]['precision'].append(score[metric].precision)
            scores[metric]['recall'].append(score[metric].recall)
            scores[metric]['fmeasure'].append(score[metric].fmeasure)
    
    return scores

def print_scores(scores):
    explanation = """
=== ROUGE 점수 설명 ===
ROUGE 점수는 생성된 요약문의 품질을 평가하는 지표입니다.

1. ROUGE-1: 단일 단어(unigram) 기반 평가
2. ROUGE-2: 두 단어 연속(bigram) 기반 평가
3. ROUGE-L: 최장 공통 부분수열 기반 평가

각 점수의 의미:
- Precision(정밀도): 생성된 요약문의 단어 중 원본 요약문과 일치하는 비율
- Recall(재현율): 원본 요약문의 단어 중 생성된 요약문에 포함된 비율
- F-measure(F1 점수): Precision과 Recall의 조화평균

점수는 0~1 사이이며, 1에 가까울수록 원본 요약문과 더 유사함을 의미합니다.
"""
    result = [explanation, "\n=== ROUGE 점수 결과 ==="]
    
    for metric, values in scores.items():
        result.append(f"\n{metric}:")
        for key, value in values.items():
            avg_value = np.mean(value)
            result.append(f"  {key}: {avg_value:.4f}")
    
    result_text = '\n'.join(result)
    print(result_text)
    return result_text

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
    # 데이터 로드
    data = load_data()
    
    # 테스트 샘플 수 설정
    test_samples = min(3, len(data))
    
    # 결과 파일 초기화
    with open('output.txt', 'w', encoding='utf-8') as f:
        f.write("=== 텍스트 요약 평가 결과 ===\n")
        f.write("\n요약 방식: 추출적 요약 (TextRank 알고리즘 기반)\n")
        f.write(f"평가 샘플 수: {test_samples}\n\n")
    
    # 요약 생성 및 평가
    original_summaries = []
    generated_summaries = []
    test_texts = []
    
    print("\n=== 요약 생성 및 평가 중 ===")
    for i in range(test_samples):
        text = data['text'].iloc[i]
        original_summary = data['headlines'].iloc[i]
        generated_summary = extractive_summary(text)
        
        test_texts.append(text)
        original_summaries.append(original_summary)
        generated_summaries.append(generated_summary)
        
        output_text = (
            f"\n[샘플 {i+1}]\n"
            f"원문 (앞부분): {text[:200]}...\n"
            f"원본 요약: {original_summary}\n"
            f"생성된 요약: {generated_summary}\n"
            f"---\n"
        )
        
        # 파일에 저장
        with open('output.txt', 'a', encoding='utf-8') as f:
            f.write(output_text)
        
        # 콘솔에 출력
        print(output_text)
    
    # ROUGE 점수 계산 및 출력
    scores = evaluate_summaries(original_summaries, generated_summaries)
    result_text = print_scores(scores)
    
    # ROUGE 점수 결과를 파일에 추가
    with open('output.txt', 'a', encoding='utf-8') as f:
        f.write("\n" + result_text)

if __name__ == "__main__":
    main() 
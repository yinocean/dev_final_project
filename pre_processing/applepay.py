import pandas as pd
import re
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import ast
import os

# 현재 스크립트 파일이 위치한 디렉토리를 기준으로 경로 설정
current_directory = os.path.dirname(os.path.abspath(__file__))

# CSV 파일 읽기 (현재 디렉토리 기준 상대 경로 사용)
file_path = os.path.join(current_directory, '잇섭_preprocess_final_0830.csv')
df = pd.read_csv(file_path)

# 불용어 사전 읽기 (현재 디렉토리 기준 상대 경로 사용)
stopwords_file_path = os.path.join(current_directory, '불용어사전_240807.csv')
stopwords_df = pd.read_csv(stopwords_file_path, encoding='cp949', header=None, names=['word'])
stopwords_list = stopwords_df['word'].tolist()  # 불용어 사전에서 단어 리스트 추출

# Kiwi와 Stopwords 객체 초기화
kiwi = Kiwi()
stopwords = Stopwords()
stopwords.add(stopwords_list)  # 불용어 사전에 사용자가 추가한 불용어 리스트 포함

def compress_punctuation(text):
    text = re.sub(r'!+', '!', text)  # 연속된 느낌표를 하나로 압축
    text = re.sub(r'\?+', '?', text)  # 연속된 물음표를 하나로 압축
    text = re.sub(r'[ㅜㅠ]', '', text)  # 'ㅜ'와 'ㅠ' 제거
    return text

# 이중 리스트 형태의 comment_tag에서 '애플페이' 또는 '애페'가 포함된 텍스트 추출하는 함수
def extract_relevant_tags(comment_tags):
    try:
        tags_list = ast.literal_eval(comment_tags)  # 문자열을 실제 리스트로 변환
        relevant_texts = []
        for tag_tuple_list in tags_list:
            if any('애플페이' in word or '애페' in word for word, pos in tag_tuple_list):
                relevant_texts.append(' '.join([word for word, pos in tag_tuple_list]))
        return ' '.join(relevant_texts) if relevant_texts else None
    except (SyntaxError, ValueError):
        return None

# comment_tag에서 '애플페이' 또는 '애페'가 포함된 텍스트 추출
filtered_df = df.copy()
filtered_df['extracted_text'] = filtered_df['comment_tag'].apply(extract_relevant_tags)

# NaN 값 제거 (필터링된 텍스트가 없는 경우 제외)
filtered_df = filtered_df.dropna(subset=['extracted_text'])

# comment_text에 대해 전처리 적용
filtered_df['comment_text'] = filtered_df['comment_text'].apply(compress_punctuation)

# 'published_at'을 datetime 형식으로 변환
filtered_df['published_at'] = pd.to_datetime(filtered_df['published_at'], errors='coerce')

# 감정 분석 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("Copycats/koelectra-base-v3-generalized-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("Copycats/koelectra-base-v3-generalized-sentiment-analysis")
sentiment_classifier = TextClassificationPipeline(tokenizer=tokenizer, model=model)

# 불용어 제거 후 감정 분석 수행 함수
def analyze_sentiment_after_stopwords_removal(text):
    tokens = kiwi.tokenize(text, stopwords=stopwords)
    filtered_text = ' '.join([token.form for token in tokens])  # 불용어 제거 후 남은 텍스트
    
    # 불용어 제거 후 남은 텍스트가 비어있지 않고, 텍스트 길이가 충분할 때 감정 분석 수행
    if filtered_text.strip() and (len(filtered_text.split()) > 1):  # 텍스트가 비어있지 않고 단어가 1개 이상인 경우
        result = sentiment_classifier(filtered_text)
        label = result[0]['label']
        score = result[0]['score']
        return label, score
    
    return None, None  # 텍스트가 비어있거나 길이가 짧을 경우 None 반환

# 감정 분석 수행 (extracted_text에 대해 감정 분석)
filtered_df['applepay_sentiment_label'], filtered_df['applepay_sentiment_score'] = zip(
    *filtered_df['extracted_text'].apply(analyze_sentiment_after_stopwords_removal)
)

# 결과 CSV 파일을 같은 폴더에 저장
output_file_path = os.path.join(current_directory, 'applepay_comments_with_sentiment.csv')
filtered_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')

print(f"감정 분석이 완료되었으며, 결과가 {output_file_path}에 저장되었습니다.")

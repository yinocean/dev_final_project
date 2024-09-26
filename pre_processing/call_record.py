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

# 연속된 느낌표나 물음표를 하나로 압축하는 함수 정의
def compress_punctuation(text):
    text = re.sub(r'!+', '!', text)  # 연속된 느낌표를 하나로 압축
    text = re.sub(r'\?+', '?', text)  # 연속된 물음표를 하나로 압축
    text = re.sub(r'[ㅜㅠ]', '', text)  # 'ㅜ'와 'ㅠ' 제거
    return text

# 이중 리스트 형태의 comment_tag에서 통화녹음 관련 키워드가 포함된 텍스트 추출하는 함수
def extract_relevant_tags_call_recording(comment_tags):
    call_recording_keywords = [
        '통화녹음', '통화중녹음', '통화자동녹음', '통화내용녹음', 
        '통화시녹음', '전화녹음', '전화시녹음', '전화자동녹음'
    ]
    try:
        tags_list = ast.literal_eval(comment_tags)  # 문자열을 실제 리스트로 변환
        relevant_texts = []
        for tag_tuple_list in tags_list:
            if any(keyword in word for keyword in call_recording_keywords for word, pos in tag_tuple_list):
                relevant_texts.append(' '.join([word for word, pos in tag_tuple_list]))
        return ' '.join(relevant_texts) if relevant_texts else None
    except (SyntaxError, ValueError):
        return None

# 이중 리스트 형태의 comment_tag에서 '에이닷'이 포함된 텍스트 추출하는 함수
def extract_relevant_tags_adot(comment_tags):
    try:
        tags_list = ast.literal_eval(comment_tags)  # 문자열을 실제 리스트로 변환
        relevant_texts = []
        for tag_tuple_list in tags_list:
            if any('에이닷' in word for word, pos in tag_tuple_list):
                relevant_texts.append(' '.join([word for word, pos in tag_tuple_list]))
        return ' '.join(relevant_texts) if relevant_texts else None
    except (SyntaxError, ValueError):
        return None

# 통화녹음 관련 텍스트 추출
call_recording_df = df.copy()
call_recording_df['extracted_text'] = call_recording_df['comment_tag'].apply(extract_relevant_tags_call_recording)
call_recording_df = call_recording_df.dropna(subset=['extracted_text'])

# 에이닷 관련 텍스트 추출
adot_df = df.copy()
adot_df['extracted_text'] = adot_df['comment_tag'].apply(extract_relevant_tags_adot)
adot_df = adot_df.dropna(subset=['extracted_text'])

# comment_text에 대해 전처리 적용
call_recording_df['comment_text'] = call_recording_df['comment_text'].apply(compress_punctuation)
adot_df['comment_text'] = adot_df['comment_text'].apply(compress_punctuation)

# 'published_at'을 datetime 형식으로 변환
call_recording_df['published_at'] = pd.to_datetime(call_recording_df['published_at'], errors='coerce')
adot_df['published_at'] = pd.to_datetime(adot_df['published_at'], errors='coerce')

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

# 감정 분석 수행
call_recording_df['sentiment_label'], call_recording_df['sentiment_score'] = zip(
    *call_recording_df['extracted_text'].apply(analyze_sentiment_after_stopwords_removal)
)

adot_df['sentiment_label'], adot_df['sentiment_score'] = zip(
    *adot_df['extracted_text'].apply(analyze_sentiment_after_stopwords_removal)
)

# 결과 CSV 파일을 같은 폴더에 저장
output_file_path_call = os.path.join(current_directory, 'call_recording_comments_with_sentiment.csv')
output_file_path_adot = os.path.join(current_directory, 'adot_comments_with_sentiment.csv')
call_recording_df.to_csv(output_file_path_call, index=False, encoding='utf-8-sig')
adot_df.to_csv(output_file_path_adot, index=False, encoding='utf-8-sig')

print(f"통화녹음과 관련된 감정 분석이 완료되었으며, 결과가 {output_file_path_call}에 저장되었습니다.")
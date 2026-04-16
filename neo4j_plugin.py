# APOC 와 Graph Data Science Library 플러그인으로 커뮤니티(밀접한 노드들의 그룹) 생성 예제
import requests # 웹 페이지의 HTML 소스를 가져오기 위한 라이브러리
import pandas as pd # 데이터프레임(2차원배열)을 활용해 데이터를 효율적으로 저장 및 조작
import re # 정규표현식으로 문자열을 처리하기 위한 라이브러리
from bs4 import BeautifulSoup # HTML파싱을 통해 웹 데이터 스크래핑을 함
from konlpy.tag import Komoran # 한국어 자연어 처리
from neo4j import GraphDatabase
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",)

HEADERS = {'User-Agent': 'Mozilla/5.0'}
URI = "bolt://localhost:7687"

# 뉴스 데이터 수집
def fetch_news(date):
    url = f"https://news.naver.com/main/ranking/popularDay.nhn?date={date}"
    response = requests.get(url, headers=HEADERS) # 웹 페이지 HTML 소스 가져옴
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser') # HTML 파싱(데이터를 분해하여 원하는 구조로 만들고 정보를 추출)
    rankingnews = soup.find_all(class_ = 'rankingnews_box') # rankingnews_box는 인기 뉴스 섹션을 식별하는 HTML 클래스

    news_list = []
    for item in rankingnews: # 랭킹뉴스를 돌며 제목, 미디어, url, 날짜를 리스트에 저장한다.
        media = item.a.strong.text.strip()
        news = item.find_all(class_ = 'list_content')
        for new in news:
            news_list.append({
                'media': media,
                'src': f"https://news.naver.com/{new.a['href']}",
                'title': new.a.text.strip(),
                'date': date
            })
    return news_list

# 키워드 추출
def extract_keywords(df): #데이터 프레임을 입력받음
    komoran = Komoran()
    df['keyword'] = df['title'].apply(lambda title:', '.join([ # 한국어 제목에서 명사만 추출하고, 추출된 명사들을 ","로 연결해 keyword 컬럼에 저장
        noun for noun in komoran.nouns(title) if len(noun) > 1 # 각 뉴스 제목에서 2글자 이상의 명사만
    ]))
    return df
    
# 뉴스 제목 정리
def clean_title(title):
    return re.sub(r'[^a-zA-Z0-9ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', title)

# Neo4j에 데이터 저장
def save_to_neo4j(df):
    def add_article(tx, title, date, media, keyword):
        tx.run(
            """
            MERGE (a:News {title: $title, date: $date, media: $media, keyword: $keyword})
            """,
            title=title,
            date=date,
            media=media,
            keyword=keyword
        )
    def add_media(tx):
        tx.run(
            """
            MATCH (a:News)
            MERGE (b:NewsMedia {name: a.media})
            MERGE (a)-[r:Print]->(b)
            """
        )
    def add_keyword(tx):
        tx.run(
            """
            MATCH (a:News)
            UNWIND split(a.keyword, ', ') AS k
            MERGE (b:NewsKeyword {name: k})
            MERGE (a)-[r:Consists_of]->(b)
            """
        )

    driver = GraphDatabase.driver(URI, auth=("neo4j", "dnrgus2022@@"))
    with driver.session() as session:
        for _, row in df.iterrows():
            session.execute_write(add_article, title=row['clean_title'], date=row['date'], media=row['media'], keyword=row['keyword'])
            session.execute_write(add_media)
        session.execute_write(add_keyword)
    driver.close()


# 실행
def main(): 
    logging.info("뉴스 데이터를 수집합니다.")
    start_date, end_date = 20260401, 20260409
    all_news = []

    for date in range(start_date, end_date + 1):
        try:
            news = fetch_news(str(date))
            all_news.extend(news)
            logging.info(f"{date}의 뉴스 데이터를 수집했습니다.")
        except Exception as e:
            logging.error(f"{date}의 뉴스 데이터를 수집하는 중 오류가 발생했습니다: {e}")
        
    df = pd.DataFrame(all_news)

    logging.info("키워드 추출 중 ...")
    df = extract_keywords(df)

    logging.info("뉴스 제목 정리 중 ...")
    df['clean_title'] = df['title'].apply(clean_title)

    logging.info("Neo4j에 데이터를 저장합니다.")
    save_to_neo4j(df)

    logging.info("작업이 완료되었습니다.")

if __name__ == "__main__":
    main()
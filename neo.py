# Python neo4j 드라이버 패키지 설치 : pip install neo4j 
# 텍스터를 백터화하는 임베딩모델 설치 : pip install sentence-transformers
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer   

# 테스트 데이터 예시
texts = [
    "Neo4j is a graph database",
    "Graph databases are great for connected data.",
    "Machine learning can create embeddings for text."
]

# 모델 all-MiniLM-L6-v2 을 로드
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# 텍스트를 임베딩 벡터로 변환
embeddings = model.encode(texts)

# Neo4j 연결 설정
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "dnrgus2022@@"))

# Neo4j에 저장하는 함수 정의
def save_embeddings_to_neo4j(driver, texts, embeddings): # 임베딩된 백터를 원문텍스트와 함께 Neo4j에 저장하는 함수
    with driver.session() as session: # Neo4j 세션 열기
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            session.run( # Cypher 쿼리 실행
                """
                CREATE (n:Text {id: $id, content: $content, embedding: $embedding}) 
                """, # Text라는 레이블을 가진 노드로 저장
                id=i,
                content=text,
                embedding=embedding.tolist()
            )

save_embeddings_to_neo4j(driver, texts, embeddings) # 저장함수 호출

driver.close() # Neo4j 세션 닫기

import ollama
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 임베딩 모델 로드
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Neo4j 연결
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "dnrgus2022@@"))

# ✅ 모든 노드 삭제
with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")
    print("모든 노드 삭제 완료!")

# ✅ 예제 데이터 (산별로 분리)
data = {
    "북한산": """
        북한산은 서울 북부와 경기 고양·양주·의정부에 걸쳐 있는 최고봉 836.5m의 도심 국립공원(1983년 지정)으로, 
        백운대·인수봉·만경대 세 봉우리가 삼각을 이뤄 삼각산으로도 불리며, 역사적 명산입니다.
        북한산은 최고봉인 백운대(836.5m)를 중심으로 북쪽의 인수봉(810m), 남쪽의 만경대(799m) 세 봉우리가 나란히 솟아 있어 삼각산이라 불렸습니다.
        서울특별시와 경기도 여러 시군에 걸쳐 있는 산으로, 1983년 대한민국 15번째 국립공원으로 지정되었습니다.
        북한산은 조선 숙종 때 축성한 북한산성을 품고 있으며, 신라 진흥왕 순수비가 세워졌던 비봉이 있을 정도로 역사가 깊습니다.
    """,
    "백두산": """
        백두산(해발 2,744m)은 한반도에서 가장 높은 민족의 영산으로, 북한 양강도와 중국 지린성 경계에 위치한 활화산입니다.
        정상의 칼데라 호수인 천지와 흰 부석이 덮인 모습이 특징이며, 백두대간의 시발점이자 한민족의 성지입니다.
    """
}

# ✅ 청킹 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
)

# ✅ 산별로 메인노드 + 설명노드(엣지 포함) 저장
with driver.session() as session:
    for mountain_name, text in data.items():

        # 1. 메인 노드 생성 (name만 있는 노드)
        session.run( # name은 검색 필터 역할이라 임베딩 불필요
            "CREATE (n:Mountain {name: $name})", 
            name=mountain_name
        )
        print(f"{mountain_name} 메인 노드 생성 완료!")

        # 2. 청킹
        docs = text_splitter.create_documents([text])
        print(f"{mountain_name} 청크 수: {len(docs)}")

        # 3. 임베딩 생성
        embeddings = model.encode([doc.page_content for doc in docs])

        # 4. 설명 노드 + 엣지 생성
        for i, (doc, embedding) in enumerate(zip(docs, embeddings)):
            session.run(
                """
                MATCH (a:Mountain {name: $name})
                CREATE (a)-[:explain]->(b:Mountain {
                    id: $id,
                    content: $content,
                    embedding: $embedding
                })
                """,
                name=mountain_name,
                id=i,
                content=doc.page_content,
                embedding=embedding.tolist()
            )
        print(f"{mountain_name} 설명 노드 + 엣지 생성 완료!")

print("\n===== 모든 데이터 저장 완료! =====\n")

# 검색 쿼리 임베딩
query = "Explain 백두산"
query_embedding = model.encode([query])[0]

# 코사인 유사도 계산 함수
def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# 유사한 텍스트를 Neo4j에서 검색하는 함수
def find_similar_texts(driver, query_embedding, top_n=3):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (a:Mountain {name: '백두산'})-[:explain]->(b:Mountain)
            RETURN b.content AS content,
                   b.embedding AS embedding
            """
        )
        similarities = []
        for record in result:
            content = record["content"]
            embedding = record["embedding"]

            if embedding is None:  # None 체크
                continue

            similarity = cosine_similarity(query_embedding, embedding)
            similarities.append((content, similarity))
            similarities = sorted(similarities,
                              key=lambda x: x[1],
                              reverse=True)
        return similarities[:top_n]

# 유사도가 높은 텍스트를 기반으로 ollama 모델을 사용해 응답 생성
def generate_response(similar_texts):
    formatted_results = "\n".join([
        f"Result {i+1}: {result} (유사도: {similarity:.4f})"
        for i, (result, similarity) in enumerate(similar_texts)
    ])
    prompt = f"""
        You are an AI Assistant. Based on the following similarity search results, provide a helpful response to the user in Korean:
        Similarity Search Results:
        {formatted_results}
        Answer (Please respond in Korean):
        Example
         백터 유사도: , 문장내용:
         설명:
    """
    response = ollama.chat(
        model="gpt-oss:120b-cloud",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['message']['content']

# 유사한 텍스트 검색
similar_texts = find_similar_texts(driver, query_embedding)

# 유사도 결과 출력
print("===== 유사도 검색 결과 =====")
for i, (content, similarity) in enumerate(similar_texts):
    print(f"Result {i+1} (유사도: {similarity:.4f}): {content[:50]}...")
print("=" * 40)

# ollama 응답 출력
response = generate_response(similar_texts)
print(f"\n===== Generated Response =====\n{response}")
driver.close()
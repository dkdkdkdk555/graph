# embedding으로만 유사도 검사 시 발생하는 Haluciation 현상을 Cyper 쿼리와 같이 사용해서 개선되는 것을 보여주는 프로그램
import ollama
from neo4j import GraphDatabase # neo4j 클라이언트 드라이브 패키지
from sentence_transformers import SentenceTransformer # 텍스트를 백터로 임베딩하는 사전학습 모델
from numpy import dot # 백터 간의 내적(dot product)을 계산한다(백터간의 유사도 측정)
from numpy.linalg import norm # 백터의 크기or길이를 계산(linalg는 선형대수학 연산을 제공하는 서브모듈이다)
# 백터간의 코사인 유사도를 계산할 때, 백터의 크기를 고려하여 정규화된 값을 반환하는 데 사용
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 임베딩 모델 로드
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 예제 데이터와 청킹
# texts = """
# 북한산은 서울 북부와 경기 고양·양주·의정부에 걸쳐 있는 최고봉 836.5m의 도심 국립공원(1983년 지정)으로, 백운대·인수봉·만경대 세 봉우리가 삼각을 이뤄 ‘삼각산’으로도 불리며, 조선 시대 북한산성 축성 이후 ‘북한산(한강 북쪽의 큰 산)’으로 불리는 역사적 명산입니다. 
# 북한산은 최고봉인 백운대(836.5m)를 중심으로 북쪽의 인수봉(810m), 남쪽의 만경대(799m) 세 봉우리가 나란히 솟아 있어 예로부터 '삼각산'이라 불렸습니다. 화강암으로 이루어진 이 세 봉우리는 북한산의 상징적인 절경입니다.
# 서울특별시와 경기도 여러 시군에 걸쳐 있는 산으로, 1983년 대한민국 15번째 국립공원으로 지정되었습니다. 도심 한가운데 위치하여 많은 등산객이 찾는 서울의 대표적인 자연 휴식처이자 녹색 허파 역할을 합니다.
# 북한산은 조선 숙종 때 축성한 북한산성을 품고 있으며, 신라 진흥왕 순수비가 세워졌던 비봉이 있을 정도로 역사가 깊습니다. 서울의 진산(鎭山)으로서 예로부터 오악(五嶽) 중 하나로 꼽힐 정도로 중요하게 여겨졌습니다.
# 백두산(해발 2,744m)은 한반도에서 가장 높은 민족의 영산으로, 북한 양강도와 중국 지린성 경계에 위치한 활화산입니다. 정상의 칼데라 호수인 천지와 흰 부석이 덮인 모습이 특징이며, 백두대간의 시발점이자 한민족의 성지입니다. 
# """
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 200,
#     chunk_overlap = 20,
# )
# docs = text_splitter.create_documents([texts])
# 임베딩
# embeddings = model.encode([doc.page_content for doc in docs])
# Neo4j 연결
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "dnrgus2022@@"))

# mountain이라는 라벨로 neo4j에 create
# with driver.session() as session:
#     for i, (doc, embedding) in enumerate(zip(docs, embeddings)):
#         session.run(
#             """
#             CREATE (n:Mountain {id: $id, content: $content, embedding: $embedding})
#             """,
#             id=i,
#             content=doc.page_content,
#             embedding=embedding.tolist()    
#         )

# 검색 쿼리 임베딩
query = "Explain 백두산"
query_embedding = model.encode([query])[0] # encode는 여러 문장을 처리하는 함수기에 리스트로 감싸서 전달하고
    # 단일 쿼리이므로 첫번째 결과를 사용한다.

# 코사인 유사도 계산 함수
def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# 유사한 텍스트를 Neo4j에서 검색하는 함수
def find_similar_texts(driver, query_embedding, top_n=3):
    with driver.session() as session:
        result = session.run( 
            """
            MATCH (n:Mountain)
            WHERE n.content CONTAINS '백두산'
            RETURN n.content AS content,
                n.embedding AS embedding
            """
        )
        similarities = []
        for record in result:
            content = record["content"]
            embedding = record["embedding"]
            similarity = cosine_similarity(query_embedding, embedding) # 코사인 유사도 검사
            similarities.append((content, similarity)) # 각 검색 결과별 입력쿼리와의 유사도와 내용을 similarities 배열에 저장
            similarities = sorted(similarities, # similarities = [("문장1", 0.72), ("문장2", 0.91), ("문장3", 0.45)]
                              key=lambda x: x[1], # 정렬 기준 : x=각 리스트의 요소, x[1]=각 요소의 두번째 값
                              reverse=True) # 내림차순, 유사도 높은순
        return similarities[:top_n] # 상위3개만 반환

# 유사도가 높은 텍스트를 기반으로 ollama 모델을 사용해 응답 생성
def generate_response(similar_texts):
    formatted_results = "\n".join([f"Result {i+1}: {result} (유사도: {similarity:.4f})" for i, (result, similarity) in enumerate(similar_texts)])
    prompt = f"""
        You are an AI Assistant. Based on the follwing similarity search results, provide a helpful response to the user in Korean :
        Similarity Search Results:
        {formatted_results}
        Answer (Please resonsd in Korean):
        Example
         백터 유사도 :, 문장내용: 
         설명 : 
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
response = generate_response(similar_texts)
print(f"Generated Response: {response}")
driver.close()


# FAISS + LangChain 으로 문서 유사도 검색 수행하는 프로그램 예제
from langchain_community.vectorstores import FAISS # 백터를 FAISS백터스토어에 저장하고 관리
from langchain_text_splitters import CharacterTextSplitter # 구분자 하나로만 텍스트를 청킹(자름)
from langchain_ollama import OllamaEmbeddings # 텍스트 데이터를 백터로 임베딩 +  유사도 검색 수행
from langchain_core.documents import Document # 텍스트 데이터를 담는 기본 데이터 컨테이너

embeddings = OllamaEmbeddings(model="bge-m3")

documents = [
    Document(page_content="Sweet potato is a perennial root vegetable of the morning glory family, a cultivated crop with starchy, sweet-tasting tuberous roots. Its flowers resemble morning glories, and while it can reproduce by seed, seeds are not used when cultivating it for edible roots."),
    Document(page_content="Sweet potatoes can be enjoyed in a variety of dishes such as candied sweet potatoes, gratin, and latte. Popular recipes include cheese-baked sweet potato, sweet potato salad, and sweet potato candy. They are also popular as quick meals, such as baking in an air fryer or making cheese gratin in the microwave in just 3 minutes."),
    Document(page_content="The potato is a perennial plant of the nightshade family and is one of the world's four major crops along with rice, wheat, and corn. Together with sweet potatoes and corn, it is a representative famine relief crop that has saved humanity from the fear of starvation.")
]

# 1. 청킹 - CharacterTextSplitter가 Document를 작은 조각으로 분할
text_splitter = CharacterTextSplitter(separator="", chunk_size=150, chunk_overlap=15)
split_documents = text_splitter.split_documents(documents)

# 2. 임베딩 + 저장 동시에 - from_documents()가 내부적으로 OllamaEmbeddings 호출해서
#    벡터 변환 후 FAISS 인덱스에 저장까지 한번에 처리
faiss_index = FAISS.from_documents(split_documents, embeddings) # FAISS 백터 스토어 생성

# 3. 유사도 검색 - 쿼리도 동일한 임베딩 모델로 변환 후 FAISS가 가장 가까운 벡터 검색
# 검색 쿼리 임베딩 및 유사도 검색
query = "How to cook sweet potato"
results = faiss_index.similarity_search_with_score(query, k=3) # k는 가져오는 가장 유사한 결과의 갯수

# 검색결과 출력 (score 낮을수록 유사도 높음)
for doc, score in results:
    print(f"score: {score:.4f} | {doc.page_content[:50]}")




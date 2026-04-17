# Streamlit을 이용한 개인 문서관리RAG 시스템
# 문서 업로드 -> 문서내용 임베딩 -> 챗봇 응답시 백터db 참조하여 응답
import streamlit as st
import os # os 관련된 작업 수행
from dotenv import load_dotenv # .env 파일에 저장된 환경변수를 python 코드로 불러옴
from langchain_community.vectorstores import FAISS # 백터를 FAISS백터스토어에 저장하고 관리
from langchain_text_splitters import CharacterTextSplitter # 구분자 하나로만 텍스트를 청킹(자름)
from langchain_ollama import OllamaEmbeddings, ChatOllama # 텍스트 데이터를 백터로 임베딩 +  유사도 검색 수행
import ollama  # 질문에 대한 답변을 생성하는 언어모델
from langchain_core.messages import ChatMessage, HumanMessage, SystemMessage # LM에서 사용할 메시지 유형을 정의하는 스키마, 사용자 질문과 시스템 프롬프트를 LM에게 전달하기 위해 사용
from langchain_core.callbacks.base import BaseCallbackHandler # LM의 응답을 실시간으로 처리할 수 있도록 지원하는 콜백 핸들러를 정의하는 클래스, LM에서 생성된 토큰을 실시간으로 웹UI에 표시하기 위해 사용
from pdfminer.high_level import extract_text # PDF파일에서 텍스트를 추출하는데 사용되는 라이브러리

# 환경변수 로드
load_dotenv()
# 임베딩모델 초기화
embeddings = OllamaEmbeddings(model="bge-m3")

# 마크다운 스트리밍 핸들러 클래스
class MarkdownStreamHandler(BaseCallbackHandler):
    """
        Streamlit 마크다운 컨테이너에 생성된 토큰을 실시간으로 스트리밍하는 사용자 정의 핸들러
        :LLM이 답변을 생성할 때 토큰(단어 조각)이 하나씩 나올 때마다 화면을 실시간으로 업데이트합니다.

        LLM 생성: "안" → "녕" → "하" → "세" → "요"
                ↓       ↓       ↓       ↓       ↓
            화면:     "안"  "안녕"  "안녕하"  "안녕하세"  "안녕하세요"
        on_llm_new_token()은 LangChain이 토큰 생성 시 자동으로 호출하는 콜백입니다.
        enerated_content에 토큰을 계속 누적하면서 매번 화면을 다시 그립니다.
    """
    def __init__(self, output_container, initial_content=""):
        #외부에서 전달된 출력컨테이너 객체를 저장 = #Streamlit의 출력 컨테이너 객체, 마크다운을 표시할 대상
        self.output_container = output_container
        # 누적된 텍스트 데이터를 저장 = # 스트리밍 시작 시 초기화된 상태의 텍스트
        self.generated_content = initial_content 
    def on_llm_new_token(self, token: str, **kwargs) -> None: # -> None 은 이 함수가 아무것도 반환하지 않는다라는 뜻 -> int: #int 반환
        self.generated_content += token
        self.output_container.markdown(self.generated_content) # 마크다운 컨테이너(=output_container) = Streamlit의 UI요소
    

# PDF 텍스트 추출 함수
def extract_text_from_pdf(file):
    """ pdfminer를 사용하여 PDF 파일에서 텍스트를 추출. """
    try:
        return extract_text(file)
    except Exception as e:
        st.error(f"PDF에서 텍스트 추출 중 오류가 발생했습니다: {e}")
        return ""

# PDF [청킹 + 임베딩 + 백터db저장] 함수
def handle_upload_file(file):
    """ 업로드된 PDF 파일을 처리하고 백터 스토어 준비. """
    if not file:
        return None, None
    # 파일 유형에 따라 텍스트 추출
    document_text = extract_text_from_pdf(file) if file.type == "application/pdf" else ""

    if not document_text:
        st.error("업로드된 PDF 파일에서 텍스트를 추출할 수 없습니다.")
        return None, None
    
    # 문서 청킹
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    document_chunks = text_splitter.create_documents([document_text])
    st.info(f"{len(document_chunks)}개의 문서 단락이 생성되었습니다.")

    # 유사도 검색을 위한 백터 스토어 생성 + 임베딩
    vectorstore = FAISS.from_documents(document_chunks, embeddings)
    return vectorstore, document_chunks

# RAG를 사용한 응답 생성 함수
def get_rag_response(user_query, vectorstore, callback_handler):
    """ 검색된 문서를 기반으로 사용자 질문에 대한 답변을 생성. 문서가 없으면 일반 대화로 응답. """
    chat_model = ChatOllama(model="gpt-oss:120b-cloud", callbacks=[callback_handler])

    if not vectorstore:
        # PDF 없이 모델과 직접 대화
        direct_prompt = [
            SystemMessage(content="당신은 친절한 AI 어시스턴트입니다. 사용자의 질문에 성실하게 답변하세요."),
            HumanMessage(content=user_query)
        ]
        try:
            response = chat_model.invoke(direct_prompt)
            return response.content
        except Exception as e:
            st.error(f"응답 생성 중 오류가 발생했습니다: {e}")
            return ""

    # 가장 유사한 문서 3개 검색
    retrieved_docs = vectorstore.similarity_search(user_query, k=3)
    retrieved_text = "\n".join(f"문서 {i+1}: {doc.page_content}" for i, doc in enumerate(retrieved_docs))

    # RAG 프롬프트 생성
    rag_prompt = [
        SystemMessage(content="제공된 문서를 기반으로 사용자의 질문에 답변하세요. 정보가 없으면 '모르겠습니다.'라고 대답하세요."),
        HumanMessage(content=f"질문: {user_query}\n\n{retrieved_text}")
    ]

    try:
        response = chat_model.invoke(rag_prompt)
        return response.content
    except Exception as e:
        st.error(f"RAG 응답 생성 중 오류가 발생했습니다: {e}")
        return ""
    
# Streamlit UI
st.set_page_config(page_title="PDF 기반 Q&A챗봇")
st.title("PDF 기반 Q&A챗봇")

# 세션 상태 변수 초기화
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        ChatMessage(role="assistant", content="안녕하세요! PDF를 업로드하면 문서 기반으로, 없어도 자유롭게 대화할 수 있습니다.")
    ]


# 파일 업로드 처리
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요.", type=["pdf"])
if uploaded_file and uploaded_file != st.session_state.get("uploaded_file"): # 중복 업로드 방지
    vectorstore, document_text = handle_upload_file(uploaded_file)
    if vectorstore:
        st.session_state["vectorstore"] = vectorstore
        st.session_state["uploaded_file"] = uploaded_file

# 채팅 기록 표시
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        st.chat_message(message.role).write(message.content)

# 사용자 질문 입력
if user_query := st.chat_input("질문을 입력하세요."):
    st.session_state.chat_history.append(ChatMessage(role="user", content=user_query))
    with chat_container:
        st.chat_message("user").write(user_query)
    # 대화 기록 표시
    # 응답 생성
    with st.chat_message("assistant"):
        stream_output = MarkdownStreamHandler(st.empty())
        assistant_response = get_rag_response(user_query, st.session_state.get("vectorstore"), stream_output)
        if assistant_response:
            st.session_state.chat_history.append(ChatMessage(role="assistant", content=assistant_response))
    




    




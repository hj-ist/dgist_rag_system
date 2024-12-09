This is the RAG System that I'm implementing right now on.

Please check my Notion Site for the details : 

https://tree-pancake-9a5.notion.site/RAG-14f6cddc0b1780df8023ecd6d55ce159


목표

→ 온라인 논문 PDF의 URL를 정보로 넘겨서, 논문에 대한 Query를 넘기면 이에 대한 답변을 해주는 모델 제작

# Implementation (온라인 Paper를 Context로 받아 답변)

- https://arxiv.org/abs/2005.11401

1. PyPDF Loader 사용 https://wikidocs.net/232104

```python
def download_pdf(url, save_path="temp.pdf"):
	  response = requests.get(url)             # 
    with open(save_path, "wb") as file:
        file.write(response.content)
    return save_path

def load_pdf_from_url(url):
    pdf_path = download_pdf(url)
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    os.remove(pdf_path)  
    return documents
```

2. PDF → Vector Store (Embedding & Store)

```python
def process_pdf_and_create_vectorstore(url, vectorstore_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    if os.path.exists(vectorstore_path):
        print("Vector Store Loading")
        vectorstore = FAISS.load_local(vectorstore_path, embeddings=embeddings, allow_dangerous_deserialization=True )
    else:
        print("PDF -> Vector Store Creating Now..")
        documents = load_pdf_from_url(url)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(split_docs, embedding=embeddings)
        vectorstore.save_local(vectorstore_path)
        print(f"Vector Store has been stored in the'{vectorstore_path}'")
    
    return vectorstore
```

3. Retriever & LLM (Open Source)

```python
from transformers import pipeline

def summarize_query_with_vectorstore(vectorstore, query):
    relevant_docs = vectorstore.similarity_search(query, k=5)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    combined_text = " ".join([doc.page_content for doc in relevant_docs])
    summary = summarizer(combined_text, max_length=150, min_length=40, do_sample=False)

    return summary[0]['summary_text']
```

4. Test Code

```python
url = "https://arxiv.org/pdf/2005.11401.pdf"
vectorstore_path = "vectorstore_faiss"

# Vector Store Check
vectorstore = process_pdf_and_create_vectorstore(url, vectorstore_path)

# Query & RAG
query = input("논문에서 알고 싶은 내용을 입력하세요: ")
summary = summarize_query_with_vectorstore(vectorstore, query)

print("요약 결과:")
print(summary)
```


============================================

- Hallucination 발생
    - Query의 한국어 문제? → 영어로 질문해도 같은 문제 발생
    - 오픈소스 LLM의 한계 → OpenAI Key의 Limit이 다해 Meta의 BART/Large-CNN 사용 (타 LLM 적용예정)
    - 유사도 문서 개수 등 하이퍼파라미터 조정 → 이론 공부가 필요

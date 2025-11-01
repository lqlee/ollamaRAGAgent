# https://github.com/pixegami/rag-tutorial-v2
# pip install docling

from docling.document_converter import DocuementConverter
source = 'https://arxiv.org/pdf/24008.09869'
doc = documentConverter().convert(source).document

markdown-content = doc.export_to_markdown()
from langchain_text_splitters import MarkdownHeaderTextSplitter
headers_to_split_on = [
  ('#', 'Header 1'),
  ('##', 'Header 2'),
  ('###', 'Header 3'),
]
markdown-splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(markdown_content)

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(md_header_splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={'k':3})

from langchain_ollama import OllamaLLM
llm = OllamaLLM(model='granite4:micro')

from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
  llm = llm,
  chain_type = 'stuff',
  retriever = retriever,
  return_source_documents = True
  )
question = 'what is this document about ?'
result = qa_chain.invoke({'query': question})

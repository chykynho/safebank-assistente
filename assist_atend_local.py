# -*- coding: utf-8 -*-
"""
App RAG para atendimento automatizado com Streamlit - Versão Local (VSCode)
"""

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from IPython.display import Markdown, display
import os

# Carrega variáveis de ambiente (como GROQ_API_KEY) do arquivo .env
load_dotenv()

# Configuração da página do Streamlit
st.set_page_config(page_title="Atendimento SafeBank 🤖", page_icon="🤖")
st.title("Atendimento SafeBank")

# Configurações do modelo LLM
id_model = "deepseek-r1-distill-llama-70b"
temperature = 0.7

# Caminho local onde estão os PDFs (ajuste conforme sua máquina)
path = "/home/francisco-alves/Documentos/cursos/iaexpert/LLMs_Agentes_IA_Empresas_Negocios/documentos/atendimento_e_suporte"

### Função para carregar o modelo de linguagem grande (LLM)
def load_llm(id_model, temperature):
    llm = ChatGroq(
        model=id_model,
        temperature=temperature,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    return llm


llm = load_llm(id_model, temperature)


### Função para exibir respostas formatadas
def show_res(res):
    if "fromTo" in res:
        res = res.split("fromTo")[-1].strip()
    else:
        res = res.strip()
    display(Markdown(res))


### Função para extrair texto de PDF
def extract_text_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    doc = loader.load()
    content = "\n".join([page.page_content for page in doc])
    return content


### Função para configurar o retriever com FAISS
def config_retriever(folder_path=path):
    docs_path = Path(folder_path)

    if not docs_path.exists():
        st.error(f"A pasta '{folder_path}' não foi encontrada.")
        st.stop()

    pdf_files = [f for f in docs_path.glob("*.pdf")]

    if not pdf_files:
        st.error(f"Nenhum arquivo PDF encontrado na pasta: {folder_path}")
        st.stop()

    loaded_documents = [extract_text_pdf(str(pdf)) for pdf in pdf_files]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for doc in loaded_documents:
        if doc.strip():
            chunks.extend(text_splitter.split_text(doc))

    if not chunks:
        st.error("Nenhum chunk foi criado. Verifique se os PDFs têm conteúdo.")
        st.stop()

    embedding_model = "BAAI/bge-m3"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    try:
        vectorstore = FAISS.load_local("index_faiss", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.warning("Índice FAISS não encontrado. Criando novo índice...")
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
        vectorstore.save_local("index_faiss")

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 4})
    return retriever


### Função para configurar a chain RAG com histórico
def config_rag_chain(llm, retriever):
    # Prompt para reformular perguntas com base no histórico
    context_q_system_prompt = (
        "Given the following chat history and the follow-up question "
        "which might reference context in the chat history, formulate a standalone question "
        "which can be understood without the chat history. Do NOT answer the question, just reformulate it."
    )

    context_q_user_prompt = "Question: {input}"
    context_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", context_q_user_prompt),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=retriever, prompt=context_q_prompt
    )

    # Prompt para responder perguntas com base no contexto
    system_prompt = """Você é um assistente virtual prestativo e está respondendo perguntas gerais sobre os serviços de uma empresa.
Use os seguintes pedaços de contexto recuperado para responder à pergunta.
Se você não sabe a resposta, apenas comente que não sabe dizer com certeza.
Mas caso seja uma dúvida muito comum, pode sugerir como alternativa uma solução possível.
Mantenha a resposta concisa.
Responda em português. \n\n"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "Pergunta: {input}\n\n Contexto: {context}"),
        ]
    )

    # Cria a chain de documentos e Q&A
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return rag_chain


### Função de interação com o usuário
def chat_llm(rag_chain, input):
    st.session_state.chat_history.append(HumanMessage(content=input))

    response = rag_chain.invoke({"input": input, "chat_history": st.session_state.chat_history})

    res = response["answer"]
    res = res.split("fromTo")[-1].strip() if "fromTo" in res else res.strip()

    st.session_state.chat_history.append(AIMessage(content=res))

    return res


# Inicializa histórico da conversa
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Olá, sou o seu assistente virtual! Como posso te ajudar?"),
    ]

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Exibe histórico da conversa
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# Campo de entrada do usuário
input = st.chat_input("Digite sua mensagem aqui...")

if input is not None:
    with st.chat_message("Human"):
        st.markdown(input)

    with st.chat_message("AI"):
        if st.session_state.retriever is None:
            with st.spinner("Carregando documentos..."):
                st.session_state.retriever = config_retriever(path)

        rag_chain = config_rag_chain(llm, st.session_state.retriever)
        res = chat_llm(rag_chain, input)
        st.write(res)
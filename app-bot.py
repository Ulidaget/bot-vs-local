import streamlit as st
import boto3
from langchain_community.chat_models import BedrockChat
from langchain_community.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

# Configuración de AWS
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")


folder_path="/tmp/"

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"  # Cambia esto a tu región de AWS
)

# Función para inicializar el modelo Claude 3
def init_claude():
    llm = BedrockChat(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",  # Asegúrate de usar el ID correcto
        client=bedrock_runtime,
        model_kwargs={"max_tokens": 4096, "temperature": 0.5, "top_p": 1}
    )
    return llm

# Función para cargar el vector store FAISS
def load_vectorstore():
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        client=bedrock_runtime
    )
    # vectorstore = FAISS.load_local("vectorstore_4", embeddings, allow_dangerous_deserialization=True)
    vectorstore = FAISS.load_local(
        index_name="index",
        folder_path = folder_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore

def load_index():
    s3_client.download_file(Bucket=BUCKET_NAME, Key="index.faiss", Filename=f"{folder_path}index.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="index.pkl", Filename=f"{folder_path}index.pkl")

# Inicializar el chatbot
@st.cache_resource
def init_chatbot():
    llm = init_claude()
    vectorstore = load_vectorstore()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return qa_chain


load_index()

# dir_list = os.listdir(folder_path)
# st.write(f"Files and Directories in {folder_path}")
# st.write(dir_list)


# Interfaz de Streamlit
st.title("Honne Bot local vector - store")

# Inicializar el chatbot
qa_chain = init_chatbot()

# Mantener el historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# # Entrada del usuario
# if prompt := st.chat_input("Escribe tu mensaje aquí"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Generar respuesta
#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         response = qa_chain({"question": prompt})
#         full_response = response['answer']
#         message_placeholder.markdown(full_response)
    
#     # Guardar la respuesta en el historial
#     st.session_state.messages.append({"role": "assistant", "content": full_response})
# Entrada del usuario
if prompt := st.chat_input("Escribe tu mensaje aquí"):
    # Contexto adicional o instrucciones claras
    enhanced_prompt = (
        f"Eres un asistente virtual experto en marketing con la informacion de Honne Services. "
        f"Por favor, responde con información detallada y precisa. "
        f"\n\nPregunta del usuario: {prompt}"
    )

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar respuesta
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = qa_chain({"question": enhanced_prompt})
        full_response = response['answer']
        message_placeholder.markdown(full_response)
    
    # Guardar la respuesta en el historial
    st.session_state.messages.append({"role": "assistant", "content": full_response})
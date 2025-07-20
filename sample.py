import streamlit as st
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAEvalChain
from langchain_groq import ChatGroq

#LLM and key loading function
# LLM and key loading function
def load_LLM(groq_api_key):
    """Load LLaMA model from Groq API."""
    llm = ChatGroq(
        model="llama3-70b-8192",  # or "llama3-8b-8192" if you want smaller model
        groq_api_key=groq_api_key,
        temperature=0.7
    )
    return llm
from langchain_community.embeddings import FastEmbedEmbeddings
# Use FastEmbed (open-source embeddings) instead of OpenAIEmbeddings
embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def generate_response(
    uploaded_file,
    groqai_api_key,
    query_text,
    response_text
):
    #format uploaded file
    documents = [uploaded_file.read().decode()]
    
    #break it in small chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    texts = text_splitter.create_documents(documents)
    embeddings = embedding_model 
    

    
    # create a vectorstore and store there the texts
    db = FAISS.from_documents(texts, embeddings)
    
    # create a retriever interface
    retriever = db.as_retriever()
    
    # create a real QA dictionary
    real_qa = [
        {
            "question": query_text,
            "answer": response_text
        }
    ]
    
    # regular QA chain
    qachain = RetrievalQA.from_chain_type(
        llm=load_LLM(groqai_api_key=groqai_api_key),
        chain_type="stuff",
        retriever=retriever,
        input_key="question"
    )
    
    # predictions
    predictions = qachain.apply(real_qa)
    
    # create an eval chain
    eval_chain = QAEvalChain.from_llm(
       llm=load_LLM(groqai_api_key=groqai_api_key)
    )
    # have it grade itself
    graded_outputs = eval_chain.evaluate(
        real_qa,
        predictions,
        question_key="question",
        prediction_key="result",
        answer_key="answer"
    )
    
    response = {
        "predictions": predictions,
        "graded_outputs": graded_outputs
    }
    
    return response

st.set_page_config(
    page_title="Evaluate a RAG App"
)
st.title("Evaluate a RAG App")

with st.expander("Evaluate the quality of a RAG APP"):
    st.write("""
        To evaluate the quality of a RAG app, we will
        ask it questions for which we already know the
        real answers.
        
        That way we can see if the app is producing
        the right answers or if it is hallucinating.
    """)

uploaded_file = st.file_uploader(
    "Upload a .txt document",
    type="txt"
)

query_text = st.text_input(
    "Enter a question you have already fact-checked:",
    placeholder="Write your question here",
    disabled=not uploaded_file
)

response_text = st.text_input(
    "Enter the real answer to the question:",
    placeholder="Write the confirmed answer here",
    disabled=not uploaded_file
)

result = []
with st.form(
    "myform",
    clear_on_submit=True
):
    groqai_api_key = st.text_input(
        "groqAI API Key:",
        type="password",
        disabled=not (uploaded_file and query_text)
    )
    submitted = st.form_submit_button(
        "Submit",
        disabled=not (uploaded_file and query_text)
    )
    if submitted and groqai_api_key.startswith("sk-"):
        with st.spinner(
            "Wait, please. I am working on it..."
            ):
            response = generate_response(
                uploaded_file,
                groqai_api_key,
                query_text,
                response_text
            )
            result.append(response)
            del groqai_api_key
            
if len(result):
    st.write("Question")
    st.info(response["predictions"][0]["question"])
    st.write("Real answer")
    st.info(response["predictions"][0]["answer"])
    st.write("Answer provided by the AI App")
    st.info(response["predictions"][0]["result"])
    st.write("Therefore, the AI App answer was")
    st.info(response["graded_outputs"][0]["results"])


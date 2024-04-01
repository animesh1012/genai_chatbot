import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import transformers
from transformers import pipeline
from transformers import AutoTokenizer
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.llm import LLMChain
from langchain_community.llms import VLLM
import torch
import parsers
from guardrails.check_input_query import validate_query

st.set_page_config(page_title="CCMT Councelling Assistant", layout="wide")

torch.cuda.set_device(torch.device('cuda:0'))

#Loading the model
@st.cache_resource()
def load_llm(model_name):
    # Load the locally downloaded model here
    bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype='bfloat16'
        )
    model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map = 'auto'
        )
    
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # pipe = pipeline('text-generation', model=model,max_new_tokens = 512,tokenizer=tokenizer,top_k = 10,top_p = 0.95,temperature = 0.05)
    # llm=HuggingFacePipeline(pipeline=pipe)
    llm = VLLM(
    model = model_name,
    trust_remote_code=True,  # mandatory for hf models
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    temperature=0.01,
    gpu_memory_utilization=0.80,
)
    return llm

def load_model():
    with st.spinner("Loading the chat model"):
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        llm_model = load_llm(model_name)
    return llm_model
llm_model = load_model()



#Creating Chain
#@st.cache_resource()
def get_conversational_chain():
    # prompt_template = """
    # Answer the question as detailed as possible from the provided context, make sure to provide all the details,
    # if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    # Context:\n {context}\n
    # Question:\n {question}\n
    
    # Answer:
    # """
    
    prompt_template = """You are a friendly user chatting agent who replies to user queries in a polite and humble way.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Ask for more information from the user, if required. Do not jump to conclusion
    Answer only those questions related to CCMT Councelling
    Do not answer mathematical, political questions

    If the question is not related to CCMT Councelling, then just reply - "I cannot help you with that" as the only response and do not answer further
    If required, use the following pieces of context to answer the question. 
    {context}
    Current conversation:
    [INST] Do not answer mathematical or current events or political questions[/INST]
    Sure. I do not answer such questions
    {history}
    [INST] {input} [/INST]"""

    # model_name = "meta-llama/Llama-2-7b-chat-hf"
    #model_name = "Llama-2-13b-chat-hf-quantized"
    # model = load_llm(model_name)
    output_parser = parsers.OutputParser()
    prompt = PromptTemplate(template=prompt_template,input_variables=["context","history","input"],output_parser=output_parser)
    chain = LLMChain(llm = llm_model,prompt=prompt,output_parser=output_parser)
    
    return chain


#Chat Session Code
#Create session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages=[]

#Visualization of entire chat session
header = st.container()
header.title('Hi!! I am an Assistant for CCMT Councelling \n you may ask your queries here ')
header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)
st.markdown(
    """
<style>
    div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
        position: sticky;
        top: 1.875rem;
        background-color: white;
        z-index: 999;
    }
    .fixed-header {
        border-bottom: 1px solid black;
    }
</style>
    """,
    unsafe_allow_html=True
)
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

#Recording Last 5 chats in history string
max_buffer=5
history=''
for message in st.session_state.messages[-2*max_buffer:]:
    if message['role']=='user':
        content_str = '[INST] '+message['content']+' [/INST]\n'
    elif message['role']=='assistant':
        content_str = message['content']+'\n'
    history += content_str
# print("history :::::::::::::::::::\n", history)
    
#Generate Response
def find_answer(user_question):
    model_name='sentence-transformers/all-mpnet-base-v2'
    embedding_fn = HuggingFaceEmbeddings(model_name=model_name)
    new_db = FAISS.load_local("faiss_index",embedding_fn)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    
    # response = chain(
    #     {'input_documents':docs,'question':user_question},return_only_outputs=True
    # )
    guardrail_message = validate_query(user_question,llm_model)
    if(guardrail_message=="Reject"):
        response = "I cannot answer to your question because I'm programmed to assist only with CCMT Councelling related questions."
    else:
        response = chain.run(input=user_question, context=docs,history=history)
    # print(response)
    # st.write("Reply: ",response)
    return response


#Streamlit UI
user_input = st.chat_input('Ask your query here')
if user_input:
    with st.chat_message('user'):
        st.markdown(user_input)
    st.session_state.messages.append({'role':'user', 'content':user_input})

    with st.spinner("Processing user input"):
        response_from_llm = find_answer(user_input)
    with st.chat_message('assistant'):
        st.markdown(response_from_llm)
    st.session_state.messages.append({'role':'assistant', 'content':response_from_llm})


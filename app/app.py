import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

st.set_page_config(page_title="Assistant COVID-19", page_icon="🏥", layout="wide")

st.markdown("""
<style>
.chat-message { padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; }
.ai-message { background-color: #f0f2f6; }
.user-message { background-color: #e1f5fe; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        "PrunaAI/ruslanmv-Medical-Llama3-8B-bnb-4bit-smashed",
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config
    )
    
    model = PeftModel.from_pretrained(
        model,
        "./model_output/medical-llama-covid-french-finetuned/checkpoint-1118"
    )
    
    tokenizer = AutoTokenizer.from_pretrained("ruslanmv/Medical-Llama3-8B")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

@st.cache_resource
def create_vector_store():
    loader = DirectoryLoader(
        "./rag_data",
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Reduced chunk size for better context management
        chunk_overlap=50,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

def get_relevant_context(vector_store, question, k=2):  # Reduced number of retrieved documents
    docs = vector_store.similarity_search(question, k=k)
    context = "\n".join([doc.page_content for doc in docs])
    return context

def truncate_text(text, tokenizer, max_length):
    """Truncate text to fit within max_length tokens."""
    tokens = tokenizer.encode(text, truncation=False)
    if len(tokens) > max_length:
        tokens = tokens[:max_length - 100]  # Leave room for generation
        text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text

def generate_response(model, tokenizer, question, context):
    # Calculate available length for context
    max_length = 512
    prompt_template = f"Question: {question}\n\nRéponse basée sur le contexte suivant:\n{context}\n\nRéponse:"
    
    # Truncate context if needed
    truncated_prompt = truncate_text(prompt_template, tokenizer, max_length)
    
    inputs = tokenizer(truncated_prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,  # Control the length of generated response
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part
    response_parts = response.split("Réponse:")
    if len(response_parts) > 1:
        return response_parts[-1].strip()
    return response.strip()

def main():
    st.title("🏥 Assistant COVID-19")
    st.markdown("### Posez vos questions sur le COVID-19")

    with st.sidebar:
        st.header("ℹ️ À propos")
        st.markdown("""
        ### Notre Mission
        Fournir des informations fiables et actualisées sur le COVID-19, basées sur les dernières données scientifiques.

        ### Expertise
        - Informations validées par des experts médicaux
        - Sources officielles (OMS, autorités sanitaires)
        - Mises à jour régulières
        - Base de connaissances locale (PDFs)

        ### Avertissement Important
        ⚠️ Cet assistant virtuel est un outil d'information générale. Pour tout avis médical personnalisé, consultez un professionnel de santé.

        ### Fonctionnalités
        - Questions-réponses en temps réel
        - Informations sur les symptômes
        - Conseils de prévention
        - Actualités sur les variants
        
        ### Contact
        Pour toute question technique ou suggestion :
        📧 support@covid-assistant.com
        """)

    try:
        model, tokenizer = load_model()
        vector_store = create_vector_store()
        st.success("✅ Modèle et base de connaissances chargés avec succès")
    except Exception as e:
        st.error(f"❌ Erreur de chargement: {str(e)}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.markdown(f"""
        <div class="chat-message {'ai-message' if message['role']=='assistant' else 'user-message'}">
        <b>{'Assistant' if message['role']=='assistant' else 'Vous'}:</b> {message['content']}
        </div>
        """, unsafe_allow_html=True)

    with st.form(key="chat_form"):
        user_input = st.text_input("Votre question:")
        submit_button = st.form_submit_button("Envoyer")

    if submit_button and user_input:
        with st.spinner("Recherche et génération de la réponse..."):
            try:
                context = get_relevant_context(vector_store, user_input)
                response = generate_response(model, tokenizer, user_input, context)
                
                st.markdown(f"""
                <div class="chat-message user-message">
                <b>Vous:</b> {user_input}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="chat-message ai-message">
                <b>Assistant:</b> {response}
                </div>
                """, unsafe_allow_html=True)
                
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Une erreur est survenue: {str(e)}")

if __name__ == "__main__":
    main()
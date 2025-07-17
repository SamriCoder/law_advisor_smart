import os
import json
import hashlib
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import time

# Add this at the top after imports to disable MongoDB connections
os.environ["MONGODB_URI"] = ""
os.environ["MONGO_URI"] = ""

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not all([OPENAI_API_KEY, PINECONE_API_KEY, GOOGLE_API_KEY]):
    print(f"Missing API keys:")
    print(f"OPENAI_API_KEY: {'✓' if OPENAI_API_KEY else '✗'}")
    print(f"PINECONE_API_KEY: {'✓' if PINECONE_API_KEY else '✗'}")
    print(f"GOOGLE_API_KEY: {'✓' if GOOGLE_API_KEY else '✗'}")
    raise ValueError("Missing required API keys. Set OPENAI_API_KEY, PINECONE_API_KEY, and GOOGLE_API_KEY environment variables.")

# Debug: Print first few characters of API key (safely)
print(f"Using Pinecone API key: {PINECONE_API_KEY[:10]}...")

# Set environment variables
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

def is_legal_query(query: str) -> bool:
    blocked_keywords = [
        "how to escape", "how to avoid", "bypass", "trick", "fake", "hack",
        "delete evidence", "hide crime", "exploit law", "evade", "bribe",
        "how to not get caught", "kill", "rape", "murder", "illegal", "piracy",
        "joke", "date me", "play game", "love", "funny", "story", "song",
        "chat with me", "your name", "who created you", "what's your age"
    ]
    query = query.lower()
    return not any(keyword in query for keyword in blocked_keywords)

def load_documents(pdf_paths):
    docs = []
    for path in pdf_paths:
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping...")
            continue
        try:
            loader = PyPDFLoader(path)
            pages = loader.load_and_split()
            docs.extend(pages)
            print(f"Loaded {len(pages)} pages from {path}")
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return docs

def split_documents(documents, chunk_size=2000, chunk_overlap=400):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # Much larger chunks
        chunk_overlap=chunk_overlap,  # More overlap
        separators=[
            "\n\nSection",
            "\n\nClause", 
            "\n\nChapter",
            "\n\n",
            "\n",
            ".",
            " "
        ],
        keep_separator=True  # Keep section headers
    )
    return splitter.split_documents(documents)

# Proper Pinecone initialization with new API
def init_pinecone(index_name="lawdb"):
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        existing_indexes = pc.list_indexes()
        index_names = [idx.name for idx in existing_indexes]
        
        if index_name in index_names:
            print(f"✅ Index '{index_name}' already exists. Using existing index.")
            index = pc.Index(index_name)
            return index
        
        print(f"Creating new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(10)
        
        index = pc.Index(index_name)
        return index
        
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        raise

# Proper vector store creation
def embed_documents(docs, index_name):
    try:
        # Use free Hugging Face embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        vectorstore = PineconeVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            index_name=index_name
        )
        
        print(f"Successfully embedded {len(docs)} documents")
        return vectorstore
        
    except Exception as e:
        print(f"Error embedding documents: {e}")
        raise

# Proper Gemini initialization
def init_gemini_llm():
    try:
        llm = GoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2,
            max_output_tokens=2048  
        )
        return llm
    except Exception as e:
        print(f"Error initializing Gemini: {e}")
        raise

def build_rag(vectorstore, llm):
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        )
        
        def get_legal_answer(query):
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            prompt = f"""You are a friendly legal advisor helping someone understand Indian criminal law. Answer in a conversational, easy-to-understand way.

Legal Context: {context}

User Question: {query}

Instructions:
1. Start with a direct, simple answer
2. Use everyday language, not complex legal jargon
3. Break down complex concepts into simple points
4. Use bullet points and numbered lists
5. Include specific section numbers but explain what they mean
6. Add practical examples when helpful
7. End with actionable advice or next steps
8. Keep it conversational and helpful
9. DO NOT use any markdown formatting like ** or __ or # - use plain text only
10. Use simple formatting with line breaks and bullet points (•) only

Respond like you're talking to a friend who needs legal help in plain text format:"""
            
            response = llm.invoke(prompt)
            # Clean up any remaining markdown formatting
            result_text = response.content if hasattr(response, 'content') else str(response)
            result_text = result_text.replace('**', '').replace('__', '').replace('##', '').replace('#', '')
            
            return {
                "result": result_text,
                "source_documents": docs
            }
        
        return type('RAGPipeline', (), {
            'invoke': lambda self, inputs: get_legal_answer(inputs["query"]),
        })()
        
    except Exception as e:
        print(f"Error building RAG pipeline: {e}")
        raise

def check_index_populated(index_name):
    """Check if the Pinecone index already has documents"""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        vector_count = stats.get('total_vector_count', 0)
        print(f"📊 Index '{index_name}' contains {vector_count} vectors")
        return vector_count > 0
    except Exception as e:
        print(f"Error checking index stats: {e}")
        return False

def get_existing_vectorstore(index_name):
    """Get existing vector store without re-embedding"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings
        )
        
        print(f"✅ Connected to existing vector store: {index_name}")
        return vectorstore
        
    except Exception as e:
        print(f"Error connecting to existing vector store: {e}")
        raise

def get_file_hash(file_paths):
    """Generate hash of PDF files to detect changes"""
    hasher = hashlib.md5()
    for path in file_paths:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                hasher.update(f.read())
    return hasher.hexdigest()

def save_processing_state(pdf_paths, index_name):
    """Save state to avoid reprocessing"""
    state = {
        "pdf_hash": get_file_hash(pdf_paths),
        "index_name": index_name,
        "processed_at": time.time(),
        "pdf_files": pdf_paths
    }
    
    with open("rag_state.json", "w") as f:
        json.dump(state, f)
    print("💾 Processing state saved")

def load_processing_state():
    """Load previous processing state"""
    try:
        with open("rag_state.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def should_reprocess(pdf_paths):
    """Check if we need to reprocess documents"""
    state = load_processing_state()
    if not state:
        return True
    
    current_hash = get_file_hash(pdf_paths)
    return current_hash != state.get("pdf_hash")

def reset_database(index_name="lawdb"):
    """Reset the database - use when you want to start fresh"""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Delete index
        if index_name in [idx.name for idx in pc.list_indexes()]:
            pc.delete_index(index_name)
            print(f"🗑️ Deleted index: {index_name}")
        
        # Delete local state
        if os.path.exists("rag_state.json"):
            os.remove("rag_state.json")
            print("🗑️ Deleted local state file")
            
        print("✅ Database reset complete")
        
    except Exception as e:
        print(f"Error resetting database: {e}")

class LegalRAGSystem:
    def __init__(self):
        self.vectorstore = None
        self.rag_pipeline = None
        self.initialized = False
    
    def initialize(self):
        """Initialize the RAG system"""
        if self.initialized:
            return
            
        try:
            pdf_paths = ["BNS.pdf", "BNSS.pdf", "BSA.pdf"]
            index_name = "lawdb"

            print("🔍 Initializing Legal RAG System...")
            
            # Initialize Pinecone first
            pinecone_index = init_pinecone(index_name)
            
            # Check if index is already populated AND files haven't changed
            if check_index_populated(index_name) and not should_reprocess(pdf_paths):
                print("✅ Database already contains embeddings and files unchanged!")
                print("🚀 Connecting to existing vector store...")
                self.vectorstore = get_existing_vectorstore(index_name)
                
            else:
                print("📄 Processing documents (first time or files changed)...")
                
                # Load and process documents
                raw_docs = load_documents(pdf_paths)
                if not raw_docs:
                    raise Exception("No documents loaded. Check PDF paths.")

                print("✂️ Splitting documents into chunks...")
                chunks = split_documents(raw_docs)
                print(f"Created {len(chunks)} chunks")

                print("🔄 Generating embeddings and storing in Pinecone...")
                print("⏳ This may take a few minutes (only happens once)...")
                self.vectorstore = embed_documents(chunks, index_name)
                
                # Save state to prevent reprocessing
                save_processing_state(pdf_paths, index_name)
                print("✅ Embeddings saved to Pinecone permanently!")

            print("🧠 Initializing Gemini LLM...")
            llm = init_gemini_llm()

            print("⚙️ Building RAG system...")
            self.rag_pipeline = build_rag(self.vectorstore, llm)
            
            self.initialized = True
            print("✅ Legal RAG System initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing RAG system: {e}")
            raise
    
    def is_legal_query(self, query: str) -> bool:
        """Check if query is a valid legal question"""
        return is_legal_query(query)
    
    def get_legal_answer(self, query: str):
        """Get legal answer for the query"""
        if not self.initialized:
            raise Exception("RAG system not initialized. Call initialize() first.")
        
        return self.rag_pipeline.invoke({"query": query})

if __name__ == "__main__":
    try:
        pdf_paths = ["BNS.pdf", "BNSS.pdf", "BSA.pdf"]
        index_name = "lawdb"

        print("🔍 Checking existing setup...")
        
        # Initialize Pinecone first
        pinecone_index = init_pinecone(index_name)
        
        # Check if index is already populated AND files haven't changed
        if check_index_populated(index_name) and not should_reprocess(pdf_paths):
            print("✅ Database already contains embeddings and files unchanged!")
            print("🚀 Connecting to existing vector store...")
            vectorstore = get_existing_vectorstore(index_name)
            
        else:
            print("📄 Processing documents (first time or files changed)...")
            
            # Load and process documents
            raw_docs = load_documents(pdf_paths)
            if not raw_docs:
                print("❌ No documents loaded. Check PDF paths.")
                exit(1)

            print("✂️ Splitting documents into chunks...")
            chunks = split_documents(raw_docs)
            print(f"Created {len(chunks)} chunks")

            print("🔄 Generating embeddings and storing in Pinecone...")
            print("⏳ This may take a few minutes (only happens once)...")
            vectorstore = embed_documents(chunks, index_name)
            
            # Save state to prevent reprocessing
            save_processing_state(pdf_paths, index_name)
            print("✅ Embeddings saved to Pinecone permanently!")

        print("🧠 Initializing Gemini LLM...")
        llm = init_gemini_llm()

        print("⚙️ Building RAG system...")
        rag_pipeline = build_rag(vectorstore, llm)

        print("\n🧑‍⚖️ Smart Legal Advisor is READY!")
        print("💡 Embeddings are stored permanently - no re-processing needed!")
        print("\n" + "="*60)
        print("💬 Ask me anything about Indian Criminal Law!")
        print("📝 Examples:")
        print("   • What happens if someone steals my phone?")
        print("   • Can police arrest without warrant?")
        print("   • What is the punishment for fraud?")
        print("="*60)
        print("Type 'exit' to quit, 'help' for more examples")

        while True:
            query = input("\n🤔 Your Question: ")
            
            if query.lower().strip() == "exit":
                print("\n👋 Thanks for using Legal Advisor! Stay safe!")
                break
                
            if query.lower().strip() == "help":
                print("\n📚 Example Questions You Can Ask:")
                print("• What is theft under BNS?")
                print("• Procedure for filing FIR")
                print("• Rights during police custody")
                print("• Punishment for domestic violence")
                print("• How to get bail?")
                print("• What is cybercrime?")
                continue

            if not is_legal_query(query):
                print("\n🚫 Please ask a proper legal question about Indian criminal law.")
                print("💡 Try asking about crimes, procedures, or your legal rights.")
                continue

            try:
                print("\n🔍 Searching legal database...")
                result = rag_pipeline.invoke({"query": query})
                
                print("\n" + "="*80)
                print("⚖️  LEGAL ADVICE")
                print("="*80)
                print(result["result"])
                print("="*80)
                
                if result.get("source_documents"):
                    print("\n📚 Legal References:")
                    print("-"*50)
                    
                    for i, doc in enumerate(result["source_documents"][:3], 1):
                        source = doc.metadata.get('source', 'Legal Document')
                        page = doc.metadata.get('page', 'N/A')
                        
                        # Clean up content
                        content = doc.page_content.strip()[:300]
                        content = content.replace('\n', ' ').replace('  ', ' ')
                        
                        print(f"\n📄 Reference {i}: {source} (Page {page})")
                        print(f"📝 {content}...")
                        
                    print("\n" + "-"*50)
                    print("⚠️  Disclaimer: This is general information. Consult a lawyer for specific cases.")
                        
            except Exception as e:
                print(f"\n❌ Sorry, I couldn't process your question: {e}")
                print("💡 Try rephrasing your question or ask something simpler.")

    except Exception as e:
        print(f"❌ Fatal error: {e}")

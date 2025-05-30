import os
from django.conf import settings
from langchain_community.document_loaders import UnstructuredMarkdownLoader # For now, assuming markdown
# from langchain_community.document_loaders import UnstructuredFileLoader # A more general option
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Gemini API Key - this should be loaded by settings.py from .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # This is a fallback or error, ideally settings.py should ensure GEMINI_API_KEY is loaded
    # Or this service should explicitly try to load it from Django settings if preferred
    print("Warning: GEMINI_API_KEY not found in environment. RAG service may fail.")
    # Consider: from django.conf import settings; GEMINI_API_KEY = settings.GEMINI_API_KEY
    # However, direct os.getenv is cleaner if .env is loaded early in settings.py

# Base directory for Chroma persistence - ensure settings.MEDIA_ROOT is configured
# CHROMA_PERSIST_DIRECTORY = os.path.join(settings.MEDIA_ROOT, 'chroma_db')
# Let's place it inside the project's base directory but outside media, as it's not user-facing media.
# Alternatively, it could be settings.BASE_DIR / 'chroma_db'
CHROMA_BASE_DIR = settings.BASE_DIR / 'chroma_data' # Using Path object from settings.BASE_DIR
CHROMA_PERSIST_DIRECTORY = CHROMA_BASE_DIR / 'db'

if not os.path.exists(CHROMA_PERSIST_DIRECTORY):
    os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True) # exist_ok=True is good practice

def get_vector_store_for_user(user_id: str): # Type hint for user_id
    """
    Retrieves or creates a Chroma vector store for a specific user.
    Collection names must be valid according to ChromaDB specs (e.g., 3-63 chars, no consecutive dots, etc.)
    """
    # Ensure user_id is a string and suitable for collection names (e.g., replace problematic chars if any)
    safe_user_id = str(user_id).replace("-", "_") # Example: UUIDs contain hyphens
    collection_name = f"user_{safe_user_id}_documents"
    
    # Validate collection name (basic example)
    if not (3 <= len(collection_name) <= 63):
        raise ValueError(f"Collection name '{collection_name}' length is out of bounds (3-63 chars).")
    if ".." in collection_name:
        raise ValueError(f"Collection name '{collection_name}' cannot contain consecutive dots.")
    if collection_name.startswith(".") or collection_name.endswith("."):
        raise ValueError(f"Collection name '{collection_name}' cannot start or end with a dot.")
    # Add more validation if needed based on ChromaDB docs.

    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", # This is a common model for general purpose embeddings
        google_api_key=GEMINI_API_KEY,
        task_type="retrieval_document" # Correct for document embedding for retrieval
    )
    
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=str(CHROMA_PERSIST_DIRECTORY) # Ensure persist_directory is a string
    )
    return vector_store

# LangSmith Configuration Notes:
# The prompt mentions that LangSmith env vars are set in settings.py (via .env).
# This is good. rag_service.py doesn't need to set them again.
# If they were to be sourced from Django settings directly:
# from django.conf import settings
# os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
# os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
# But os.getenv(), relying on initial load_dotenv() in settings.py, is cleaner.
# Ensure LANGCHAIN_API_KEY, LANGCHAIN_TRACING_V2, LANGCHAIN_ENDPOINT, LANGCHAIN_PROJECT
# are actually available in the environment where Django runs.
# For example, in settings.py after load_dotenv():
# LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
# LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
# LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
# LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
# if LANGCHAIN_TRACING_V2 == "true" and LANGCHAIN_API_KEY:
#    os.environ.setdefault("LANGCHAIN_TRACING_V2", LANGCHAIN_TRACING_V2)
#    os.environ.setdefault("LANGCHAIN_API_KEY", LANGCHAIN_API_KEY)
#    if LANGCHAIN_ENDPOINT:
#        os.environ.setdefault("LANGCHAIN_ENDPOINT", LANGCHAIN_ENDPOINT)
#    if LANGCHAIN_PROJECT:
#        os.environ.setdefault("LANGCHAIN_PROJECT", LANGCHAIN_PROJECT)
# This explicit os.environ.setdefault might be useful if other parts of langchain
# don't automatically pick them up from just being loaded by dotenv into Python's os.environ
# But generally, LangChain tools should respect os.getenv().

# For now, assuming .env correctly sets these for the environment Django runs in.
# The primary concern for this file is GEMINI_API_KEY for embeddings.


def load_and_preprocess_document(file_path: str):
    '''
    Loads a document from a file path and performs basic cleaning.
    Uses UnstructuredFileLoader for broader file type support.
    '''
    # Determine file type for specific loaders if necessary, or use general UnstructuredFileLoader
    # For example, if we know it's markdown:
    # from langchain_community.document_loaders import UnstructuredMarkdownLoader
    # loader = UnstructuredMarkdownLoader(file_path, mode="elements") # "elements" mode can be useful
    
    # Using UnstructuredFileLoader for general cases.
    # It tries to infer the file type. python-magic is a dependency for this.
    from langchain_community.document_loaders import UnstructuredFileLoader
    loader = UnstructuredFileLoader(file_path, mode="elements") # "elements" or "single" or "paged"

    try:
        documents = loader.load()
    except Exception as e:
        print(f"Error loading document {file_path} with UnstructuredFileLoader: {e}")
        # Consider specific handling for different file types or errors
        # For example, if it's a binary file that unstructured can't handle well by default.
        # You might need specific loaders for PDFs (PyPDFLoader), DOCX (UnstructuredWordDocumentLoader), etc.
        # if file_path.endswith(".pdf"):
        #    from langchain_community.document_loaders import PyPDFLoader
        #    loader = PyPDFLoader(file_path)
        #    documents = loader.load_and_split() # PyPDFLoader often splits by page
        # else:
        return [] # Return empty list if loading fails

    processed_docs = []
    for doc in documents:
        # Basic cleaning: strip whitespace from content and from each line
        cleaned_content = doc.page_content.strip()
        # Remove excessive newlines, keeping single newlines
        cleaned_content = "\n".join([line.strip() for line in cleaned_content.splitlines() if line.strip()])
        
        # Retain essential metadata and update page_content
        # Unstructured loaders add a lot of metadata, some might be useful.
        # 'filename', 'file_directory', 'filetype', 'category' (e.g. 'Title', 'NarrativeText')
        # We might want to keep some of this, or add our own.
        new_doc = doc.copy() # Creates a shallow copy
        new_doc.page_content = cleaned_content
        # Example: Keep only filename from metadata if desired
        # new_doc.metadata = {'source': doc.metadata.get('filename', os.path.basename(file_path))}
        
        processed_docs.append(new_doc)
    
    if not processed_docs:
        print(f"No processable content found in document: {file_path}")

    return processed_docs


def chunk_documents(documents):
    '''
    Chunks documents using RecursiveCharacterTextSplitter.
    This is a placeholder and will need refinement for Japanese semantic chunking.
    '''
    # TODO: Replace with Japanese-specific semantic chunker.
    # Considerations for Japanese:
    # - Standard tokenizers (space-based) don't work well.
    # - Need a morphological analyzer (e.g., MeCab, Janome, SudachiPy, Nagisa).
    # - Chunking should ideally happen at sentence or paragraph boundaries,
    #   or other semantically meaningful units.
    # - LangChain might have community splitters for Japanese, or one might need to be custom-built
    #   by extending TextSplitter and using a Japanese tokenizer.
    #   Example: langchain_experimental.text_splitter.SemanticChunker (uses embeddings, might be slow at ingestion)
    #   A simpler approach: split by Japanese punctuation (。, 、, 「, 」, etc.) after tokenization.

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Max characters per chunk (adjust based on model context window & typical content)
        chunk_overlap=100, # Number of characters to overlap between chunks (helps with context)
        # separators=["\n\n", "\n", "。", "、", " ", ""], # Default separators, might need Japanese specific ones
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if not chunks:
        print(f"Warning: No chunks were created from {len(documents)} documents.")
        
    return chunks


def add_document_to_user_store(user_id: str, document_model_instance):
    '''
    Processes a document file and adds its chunks to the user's vector store.
    `document_model_instance` is an instance of the `documents.models.Document`.
    `user_id` should be the string representation of the user's ID.
    '''
    if not document_model_instance.file or not hasattr(document_model_instance.file, 'path'):
        print(f"Document {document_model_instance.id} has no file or file path associated.")
        return

    file_path = document_model_instance.file.path

    # 1. Load and preprocess
    print(f"RAG Service: Loading and preprocessing document: {file_path}")
    # Ensure file exists before loading
    if not os.path.exists(file_path):
        print(f"RAG Service: File not found at path: {file_path} for document ID {document_model_instance.id}")
        # This could happen if the file was deleted after the model instance was created
        # or if MEDIA_ROOT is misconfigured leading to an incorrect path.
        # Consider raising an error or specific handling.
        return

    loaded_docs = load_and_preprocess_document(file_path)
    if not loaded_docs:
        print(f"RAG Service: No content loaded from document: {file_path}")
        return

    # 2. Chunk documents
    print(f"RAG Service: Chunking {len(loaded_docs)} document sections from {file_path}...")
    chunks = chunk_documents(loaded_docs)
    if not chunks:
        print(f"RAG Service: No chunks created for document: {file_path}")
        return
    
    # Add document ID and other relevant metadata to each chunk
    # This helps in identifying the source of the chunk during retrieval
    for chunk in chunks:
        chunk.metadata['document_id'] = str(document_model_instance.id) # Ensure UUID is string
        chunk.metadata['document_title'] = document_model_instance.title or os.path.basename(file_path)
        chunk.metadata['user_id'] = str(user_id) # Store user_id as well
        # You might also store original file_path or other Document model fields if useful
        # chunk.metadata['file_path'] = document_model_instance.file.name # Relative path from MEDIA_ROOT

    # 3. Get user's vector store
    print(f"RAG Service: Accessing vector store for user {user_id}...")
    try:
        vector_store = get_vector_store_for_user(user_id)
    except ValueError as e:
        print(f"RAG Service: Error getting vector store for user {user_id}: {e}")
        # This could be due to invalid collection name based on user_id.
        # Decide on error handling: re-raise, log, or skip.
        return
    except Exception as e: # Catch other potential errors from Chroma/embeddings
        print(f"RAG Service: Unexpected error initializing vector store for user {user_id}: {e}")
        return


    # 4. Add chunks to vector store
    # Chroma's add_documents takes a list of Document objects (our chunks are already Document objects)
    print(f"RAG Service: Adding {len(chunks)} chunks to vector store for user {user_id} (collection: {vector_store._collection.name})...")
    try:
        vector_store.add_documents(chunks)
        # Chroma with a persist_directory generally handles persistence automatically.
        # Explicitly calling persist() might be necessary for some configurations or older versions.
        # vector_store.persist() # If issues with persistence, uncommenting this might be a try.
        print(f"RAG Service: Successfully added document {document_model_instance.id} to user {user_id}'s store.")
    except Exception as e:
        # This could be an API error with Gemini, a ChromaDB operational error, etc.
        print(f"RAG Service: Error adding chunks to Chroma for user {user_id}: {e}")
        # Consider more specific error handling or re-raising.
        # If add_documents fails, the document is in the DB but not in vector store.
        # This might require a retry mechanism or marking the document as "processing_failed".
```

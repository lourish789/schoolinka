"""
Pinecone Vector Storage Script
Run this ONCE to populate your Pinecone index with PDF documents
"""

import os
import time
import re
import concurrent.futures
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Configuration
CONFIG = {
    'GOOGLE_API_KEY': os.getenv("GOOGLE_API_KEY", "AIzaSyB3N9BHeIWs_8sdFK76PU-v9N6prcIq2Hw"),
    'PINECONE_API_KEY': os.getenv("PINECONE_API_KEY", "pcsk_zRyjS_2FyS6uk3NsKW9AHPzDvvQPzANF2S3B67MS6UZ7ax6tnJfmCbLiYXrEcBJFHzcHg"),
    'INDEX_NAME': 'coach',
    'PDF_DIR': './data',
    'CHUNK_SIZE': 1500,
    'CHUNK_OVERLAP': 100,
    'BATCH_SIZE': 100,
    'EMBED_BATCH_SIZE': 50,
    'MAX_WORKERS': 4
}


def split_docs(documents, chunk_size=1500, chunk_overlap=100):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_documents(documents)


def parse_retry_wait_time(error):
    """Parse retry wait time from error messages"""
    if hasattr(error, 'response') and error.response is not None:
        retry_after = error.response.headers.get('Retry-After')
        if retry_after:
            return int(retry_after)
    
    error_message = str(error)
    match = re.search(r'(\d+)s', error_message)
    return int(match.group(1)) if match else 20


def embed_batch_with_retry(embed_model, batch_contents, max_attempts=3):
    """Embed a batch with retry logic"""
    for attempt in range(max_attempts):
        try:
            return embed_model.embed_documents(batch_contents)
        except Exception as e:
            print(f"Error embedding batch (attempt {attempt+1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                wait_time = parse_retry_wait_time(e)
                print(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                raise


def concurrent_embed_documents(embed_model, documents, batch_size=50, max_workers=4):
    """Parallelize embedding calls using ThreadPoolExecutor"""
    all_embeddings = []
    all_contents = []
    all_metadata = []
    futures = []
    
    print(f"\nEmbedding {len(documents)} document chunks...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            batch_contents = [doc.page_content for doc in batch]
            batch_metadata = [doc.metadata for doc in batch]
            
            future = executor.submit(embed_batch_with_retry, embed_model, batch_contents)
            futures.append((future, batch_contents, batch_metadata))
        
        for future, contents, metadata in tqdm(futures, desc="Embedding batches"):
            try:
                batch_embeddings = future.result()
                all_embeddings.extend(batch_embeddings)
                all_contents.extend(contents)
                all_metadata.extend(metadata)
            except Exception as e:
                print(f"Failed to embed batch: {e}")
    
    return all_embeddings, all_contents, all_metadata


def batch_upsert(index, vectors, batch_size=100):
    """Batch upsert vectors to Pinecone"""
    batches = [vectors[i:i+batch_size] for i in range(0, len(vectors), batch_size)]
    
    print(f"\nUpserting {len(vectors)} vectors in {len(batches)} batches...")
    
    for batch_number, batch in enumerate(tqdm(batches, desc="Upserting to Pinecone")):
        for attempt in range(3):
            try:
                index.upsert(vectors=batch)
                break
            except Exception as e:
                print(f"Upsert error (batch {batch_number+1}, attempt {attempt+1}): {e}")
                if attempt < 2:
                    wait_time = 10 * (attempt + 1)
                    print(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"Batch {batch_number+1} failed after 3 attempts")
                    raise


def load_pdfs_from_directory(pdf_dir):
    """Load all PDF files from directory"""
    documents = []
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}")
    
    print(f"\nFound {len(pdf_files)} PDF files")
    
    for filename in tqdm(pdf_files, desc="Loading PDFs"):
        pdf_path = os.path.join(pdf_dir, filename)
        try:
            loader = PyPDFLoader(pdf_path)
            pdf_docs = loader.load()
            
            # Add source metadata
            for doc in pdf_docs:
                doc.metadata['source'] = filename
                doc.metadata['file_path'] = pdf_path
            
            documents.extend(pdf_docs)
            print(f"  Loaded: {filename} ({len(pdf_docs)} pages)")
        except Exception as e:
            print(f"  Error loading {filename}: {e}")
    
    return documents


def main():
    """Main execution function"""
    print("="*70)
    print("AI COACH - PINECONE VECTOR STORAGE SETUP")
    print("="*70)
    
    # Validate configuration
    if not CONFIG['GOOGLE_API_KEY'] or not CONFIG['PINECONE_API_KEY']:
        raise ValueError("Missing API keys. Set GOOGLE_API_KEY and PINECONE_API_KEY")
    
    if not os.path.exists(CONFIG['PDF_DIR']):
        raise ValueError(f"PDF directory not found: {CONFIG['PDF_DIR']}")
    
    # Initialize services
    print("\n[1/6] Initializing services...")
    embed_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=CONFIG['GOOGLE_API_KEY']
    )
    print("  Google Embeddings initialized")
    
    pc = Pinecone(api_key=CONFIG['PINECONE_API_KEY'])
    print("  Pinecone client initialized")
    
    # Check/Create index
    print(f"\n[2/6] Setting up Pinecone index '{CONFIG['INDEX_NAME']}'...")
    existing_indexes = pc.list_indexes()
    existing_index_names = [index.name for index in existing_indexes.indexes]
    
    if CONFIG['INDEX_NAME'] in existing_index_names:
        print(f"  Index '{CONFIG['INDEX_NAME']}' already exists")
        
        # Ask user if they want to delete and recreate
        response = input("  Do you want to delete and recreate it? (yes/no): ").lower()
        if response == 'yes':
            print(f"  Deleting existing index...")
            pc.delete_index(CONFIG['INDEX_NAME'])
            time.sleep(10)
            print(f"  Creating new index...")
            pc.create_index(
                name=CONFIG['INDEX_NAME'],
                dimension=768,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            print(f"  Waiting for index to be ready...")
            time.sleep(60)
        else:
            print("  Using existing index (will add/update vectors)")
    else:
        print(f"  Creating new index...")
        pc.create_index(
            name=CONFIG['INDEX_NAME'],
            dimension=768,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print(f"  Waiting for index to be ready...")
        time.sleep(60)
    
    pinecone_index = pc.Index(CONFIG['INDEX_NAME'])
    print(f"  Index ready: {CONFIG['INDEX_NAME']}")
    
    # Load PDFs
    print(f"\n[3/6] Loading PDF documents from '{CONFIG['PDF_DIR']}'...")
    documents = load_pdfs_from_directory(CONFIG['PDF_DIR'])
    print(f"  Total pages loaded: {len(documents)}")
    
    # Split documents
    print(f"\n[4/6] Splitting documents into chunks...")
    print(f"  Chunk size: {CONFIG['CHUNK_SIZE']}, Overlap: {CONFIG['CHUNK_OVERLAP']}")
    docs = split_docs(
        documents, 
        chunk_size=CONFIG['CHUNK_SIZE'], 
        chunk_overlap=CONFIG['CHUNK_OVERLAP']
    )
    print(f"  Total chunks created: {len(docs)}")
    
    # Generate embeddings
    print(f"\n[5/6] Generating embeddings...")
    all_embeddings, all_contents, all_metadata = concurrent_embed_documents(
        embed_model, 
        docs, 
        batch_size=CONFIG['EMBED_BATCH_SIZE'], 
        max_workers=CONFIG['MAX_WORKERS']
    )
    print(f"  Embeddings generated: {len(all_embeddings)}")
    
    # Prepare vectors for Pinecone
    print(f"\n[6/6] Preparing and upserting vectors to Pinecone...")
    vectors_to_upsert = []
    for idx, (embedding, content, metadata) in enumerate(zip(all_embeddings, all_contents, all_metadata)):
        vector_id = f"doc_{idx}"
        vector_metadata = {
            'text': content[:1000],  # Pinecone metadata limit
            'source': metadata.get('source', 'Unknown'),
            'page': metadata.get('page', 0)
        }
        vectors_to_upsert.append((vector_id, embedding, vector_metadata))
    
    # Batch upsert
    batch_upsert(pinecone_index, vectors_to_upsert, batch_size=CONFIG['BATCH_SIZE'])
    
    # Verify
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    stats = pinecone_index.describe_index_stats()
    print(f"Total vectors in index: {stats.get('total_vector_count', 0)}")
    print(f"Index dimension: {stats.get('dimension', 0)}")
    
    print("\n" + "="*70)
    print("VECTOR STORAGE COMPLETE!")
    print("="*70)
    print("\nYour Pinecone index is ready. You can now run the main AI Coach bot.")
    print(f"Index name: {CONFIG['INDEX_NAME']}")
    print(f"Total vectors: {stats.get('total_vector_count', 0)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()

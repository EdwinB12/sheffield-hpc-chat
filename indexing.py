"""
Getting all the files and indexing them into a database
"""
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_ollama import OllamaEmbeddings

def get_urls(sitemap_path):
    """
    Get all the files in the current directory
    """
    # Load xml file for local file
    with open(sitemap_path) as f:
        sitemap = f.read()

    soup = bs4.BeautifulSoup(sitemap, features="xml")
    sitemap_tags = soup.find_all("loc")
    urls = [url.text for url in sitemap_tags]

    # Filter out urls that are not relevant
    urls = [url for url in urls if '/hpc/' in url]
    return urls

def get_files(urls):
    """
    Get all the files in the current directory
    """
    bs4_strainer = bs4.SoupStrainer(attrs={'role': 'main'})

    # Use the custom loader
    loader = WebBaseLoader(web_paths=urls, bs_kwargs={"parse_only": bs4_strainer})
    documents = loader.load()
    return documents

def split_files(documents, chunk_size, chunk_overlap):
    """
    Split the files into chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["."])
    all_splits = text_splitter.split_documents(documents)
    return all_splits

def get_ollama_embeddings_model(embeddings_model_name):
    """
    Get the embeddings model
    """
    embeddings_model = OllamaEmbeddings(model=embeddings_model_name)
    return embeddings_model

def create_vectorstore(vector_store_type, embeddings_model):
    """
    Create a vector store
    """
    if vector_store_type.lower() == "chroma":
        from langchain_chroma import Chroma
        vector_store = Chroma(embedding_function=embeddings_model)
    else:
        raise ValueError("Invalid vector store type")
    return vector_store

def add_chunks_to_vectorstore(vector_store, all_splits):
    """
    Add the chunks to the vector store
    """
    vector_store.add_documents(all_splits)
    return vector_store

def save_vectorstore(vector_store, output_path):
    """
    Save the vector store
    """
    vector_store.save(output_path)

def main(config):
    """
    Runs the indexing process
    """
    # extract values from config value into constant variables
    SITEMAP_PATH = config["sitemap_path"]
    CHUNK_SIZE = config["chunk_size"]
    CHUNK_OVERLAP = config["chunk_overlap"]
    VECTOR_STORE_TYPE = config["vector_store_type"]
    EMBEDDINGS_MODEL = config["embeddings_model"]
    VECTOR_STORE_PATH = config["vector_store_path"]

    # Get URLs from the sitemap
    urls = get_urls(SITEMAP_PATH)

    # Get files from the URLs
    documents = get_files(urls)

    # Split the files into chunks
    all_splits = split_files(documents, CHUNK_SIZE, CHUNK_OVERLAP)

    # Get the embeddings model
    embeddings_model = get_ollama_embeddings_model(EMBEDDINGS_MODEL)

    # Create the vector store
    vector_store = create_vectorstore(VECTOR_STORE_TYPE, embeddings_model=embeddings_model)

    # Add the chunks to the vector store
    vector_store = add_chunks_to_vectorstore(vector_store, all_splits)

    return vector_store

if __name__ == "__main__":
    # Load the config file. It is a yaml file
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Run the main function
    vector_store = main(config)

    # Save the vector store
    save_vectorstore(vector_store, config['vector_store_path'])
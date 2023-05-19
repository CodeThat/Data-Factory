import os
import logging
import logging.handlers
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, UnstructuredFileLoader


def index_documents():
    """Index documents.

    Args:
        None.

    Returns:
        None.

    """

    # Create the log directory if it doesn't exist
    log_dir = 'log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure logging.
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a file handler for logging to a file.
    log_file = os.path.join(log_dir, 'indexing.log')  # Specify the desired log file path within the log directory
    file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=1024, backupCount=3)
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler for logging to the console.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Define the log format.
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)

    # Set the formatter for both handlers.
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger.
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    try:
        # Load and process documents.
        logger.info('Loading and processing documents...')
        loader = DirectoryLoader('./new_articles', glob="./*.txt", loader_cls=UnstructuredFileLoader)
        documents = loader.load()
        logger.info('Loaded %d documents.', len(documents))

        # Split documents into chunks.
        logger.info('Splitting documents into chunks...')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        logger.info('Split %d documents into %d chunks.', len(documents), len(texts))

        # Create index.
        logger.info('Creating index...')
        persist_directory = 'db'
        embedding = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(documents=texts,
                                         embedding=embedding,
                                         persist_directory=persist_directory)
        vectordb.persist()
        logger.info('Created index in %s.', persist_directory)

        # Log a message indicating that the `index_documents()` function has completed successfully.
        logger.info('Indexing completed successfully.')

    except Exception as e:
        # Log an error message with exception details.
        logger.exception('Error occurred during indexing: %s', str(e))


if __name__ == '__main__':
    index_documents()

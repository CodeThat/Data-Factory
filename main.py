import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import subprocess
import dotenv
import time

dotenv.load_dotenv()


def process_llm_response(llm_answer):
    result = llm_answer['result']
    sources = llm_answer["source_documents"]

    st.write("Result:")
    st.write(result)

    st.write("Sources:")
    for source in sources:
        st.write(source.metadata['source'])


def initialize_embedding():
    embedding = OpenAIEmbeddings()
    return embedding


def track_directory_changes(directory_path):
    cmd = ["python", "tracking.py", directory_path]
    subprocess.Popen(cmd)


def track_index_build_progress():
    # Start the index building script as a separate process
    cmd = ["python", "create_db.py"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Read the output from the index building script to track the progress
    for line in iter(process.stdout.readline, b''):
        output = line.decode().strip()
        if output.startswith("Progress:"):
            progress = int(output.split(":")[1].strip())
            # Update the progress bar
            st.progress(progress)
        elif output.startswith("Error:"):
            # Print the error message
            st.error(output.split(":", 1)[1].strip())

    # Wait for the process to finish
    process.communicate()


def main():
    st.title("🦜Document Search and Indexing Tool")

    # Create sidebar column
    with st.sidebar:
        # Button for index creation
        if st.button("Create Index"):
            try:
                st.write("Index creation started.")
                with st.spinner("Building index..."):
                    track_index_build_progress()
                st.success("Index created successfully.")
            except subprocess.CalledProcessError:
                st.error("Error occurred during index creation.")

        # Button for directory tracking
        directory_path = st.text_input("Enter the directory path to track:")
        if st.button("Start Tracking"):
            track_directory_changes(directory_path)

    # Initialize embedding
    embedding = initialize_embedding()

    # Embed documents
    persist_directory = 'db'
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    # Set up the QA chain
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff",
                                           retriever=retriever,
                                           return_source_documents=True)

    # Prompt for user input
    query = st.text_input("Enter your question:")
    if st.button("Ask"):
        llm_response = qa_chain(query)
        process_llm_response(llm_response)


if __name__ == '__main__':
    main()

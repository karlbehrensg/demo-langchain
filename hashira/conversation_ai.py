from utils import DocsJSONLLoader, get_file_path
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_documents(file_path: str):
    loader = DocsJSONLLoader(file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1600, length_function=len, chunk_overlap=160
    )

    return text_splitter.split_documents(data)


def main():
    documents = load_documents(get_file_path())
    print(len(documents))
    print(documents[0])


if __name__ == "__main__":
    main()

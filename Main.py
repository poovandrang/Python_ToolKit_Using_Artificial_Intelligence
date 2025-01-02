
import streamlit as st
import os
import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

import os
import streamlit as st
import pandas as pd
from docx import Document
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import pytesseract
from PIL import Image



print("new print statement")


class PDF:

    def ChatPDF(self,text):
        # st.write(text)

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text)
        # st.write(chunks)
        # creating embeddings

        OPENAI_API_KEY = st.text_input("OPENAI API KEY", type="password")
        if OPENAI_API_KEY:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            # st.write("Embedding Created")
            # st.write(embeddings)
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            st.write("Knowledge Base created ")

            # show user input

            def ask_question(i=0):
                user_question = st.text_input("Ask a question about your PDF?", key=i)
                if user_question:
                    docs = knowledge_base.similarity_search(user_question)
                    # st.write(docs)

                    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
                    chain = load_qa_chain(llm, chain_type="stuff")
                    with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=user_question)
                        print(cb)
                    st.write(response)
                    ask_question(i + 1)

            ask_question()

    def main_pdf(self):


        hide_st_style = """
                <style>
                #mainMenue {visibility: hidden;}
                footer {visibility: hidden;}
                #header {visibility: hidden;}
                </style>
        """
        st.markdown(hide_st_style, unsafe_allow_html=True)

        # st.write(st.set_page_config)
        st.header("Ask your PDF 🤔💭")

        # uploading file
        pdf = st.file_uploader("Upload your PDF ", type="pdf")

        # extract the text
        if pdf is not None:
            option = st.selectbox("What you want to do with PDF📜", [
                "Meta Data📂",
                "Extract Raw Text📄",
                "Extract Links🔗",
                "Extract Images🖼️",
                "Make PDF password protected🔐",
                "ChatPDF💬"
            ])
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            if option == "Meta Data📂":
                st.write(pdf_reader.metadata)
            elif option == "Make PDF password protected🔐":
                pswd = st.text_input("Enter your Password", type="password")
                if pswd:
                    with st.spinner("Encrypting..."):
                        pdf_writer = PdfWriter()
                        for page_num in range(len(pdf_reader.pages)):
                            pdf_writer.add_page(pdf_reader.pages[page_num])

                        pdf_writer.encrypt(pswd)
                        with open(f"{pdf.name.split('.')[0]}_encrypted.pdf", "wb") as f:
                            pdf_writer.write(f)

                        st.success("Encryption Successful!")
                        st.download_button(
                            label="Download Encrypted PDF",
                            data=open(f"{pdf.name.split('.')[0]}_encrypted.pdf", "rb").read(),
                            file_name=f"{pdf.name.split('.')[0]}_encrypted.pdf",
                            mime="application/octet-stream",
                        )
                        try:
                            os.remove(f"{pdf.name.split('.')[0]}_encrypted.pdf")
                        except:
                            pass
            elif option == "Extract Raw Text📄":
                st.write(text)
            elif option == "Extract Links🔗":
                for page in pdf_reader.pages:
                    if "/Annots" in page:
                        for annot in page["/Annots"]:
                            subtype = annot.get_object()["/Subtype"]
                            if subtype == "/Link":
                                try:
                                    st.write(annot.get_object()["/A"]["/URI"])
                                except:
                                    pass
            elif option == "Extract Images🖼️":
                for page in pdf_reader.pages:
                    try:
                        for img in page.images:
                            st.write(img.name)
                            st.image(img.data)
                    except:
                        pass
            elif option == "PDF Annotation📝":
                for page in pdf_reader.pages:
                    if "/Annots" in page:
                        for annot in page["/Annots"]:
                            obj = annot.get_object()
                            st.write(obj)
                            st.write("***********")
                            annotation = {"subtype": obj["/Subtype"], "location": obj["/Rect"]}
                            st.write(annotation)
            elif option == "ChatPDF💬":
                PDF.ChatPDF(pdf,text)



class File:

    def process_multiple_files(self,files):
        combined_text = ""
        for uploaded_file in files:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            if file_extension == ".pdf":
                pdf_reader = PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    combined_text += page.extract_text()
            elif file_extension == ".txt":
                combined_text += uploaded_file.read().decode("utf-8")
            elif file_extension == ".xlsx":
                excel_data = pd.read_excel(uploaded_file)
                combined_text += excel_data.to_string()
            elif file_extension == ".sql":
                combined_text += uploaded_file.read().decode("utf-8")
            elif file_extension == ".docx":
                doc = Document(uploaded_file)
                for paragraph in doc.paragraphs:
                    combined_text += paragraph.text + "\n"
            elif file_extension == ".csv":
                csv_data = pd.read_csv(uploaded_file)
                combined_text += csv_data.to_string()
            # Add more file type handling here as needed
            else:
                st.warning(f"Unsupported file type: {file_extension}. Skipping.")
        # print(combined_text)
        return combined_text

    def main_file(self):

        st.header("FileQueryHub 📂🤖")

        files = st.file_uploader(
            "Upload multiple files",
            type=["pdf", "txt", "xlsx", "sql", "docx", "csv"],
            accept_multiple_files=True
        )

        if files:
            combined_text = self.process_multiple_files(files)
            # with st.expander("See explanation"):
            # st.write(combined_text)
            OPENAI_API_KEY = st.text_input("OPENAI API KEY", type="password")

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(combined_text)
            # st.write(chunks)
            # creating embeddings

            if OPENAI_API_KEY:
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                # st.write("Embedding Created")
                # st.write(embeddings)
                with st.spinner("Creating Knowledge Base..."):
                    knowledge_base = FAISS.from_texts(chunks, embeddings)
                st.success("Knowledge Base created")

                st.write("Chat with Multiple Files 🗣️📚")

                def ask_question(i=0):
                    user_question = st.text_input("Ask a question about your Document?", key=i)
                    print(user_question)
                    if user_question:
                        with st.spinner("Searching for answers..."):
                            docs = knowledge_base.similarity_search(user_question)
                            with st.expander("See docs"):
                                st.write(docs)

                            llm = OpenAI(openai_api_key=OPENAI_API_KEY)
                            chain = load_qa_chain(llm, chain_type="stuff")
                            with get_openai_callback() as cb:
                                response = chain.run(input_documents=docs, question=user_question)
                                print(cb)
                        st.write(response)
                        ask_question(i + 1)

                ask_question()


class Ocr:
    # Set the path to your Tesseract installation (adjust based on your system)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows

    def extract_text(image):
        """
        Extracts text from a PIL Image object.

        Args:
            image: A PIL Image object.

        Returns:
            The extracted text as a string.
        """
        try:
            # Convert image to RGB mode if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Improve image quality for better OCR (optional)
            # img = img.convert('L') # Convert to grayscale
            # img = img.filter(ImageFilter.SHARP) # Sharpen the image

            # Extract text from the image
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            print(f"An error occurred: {e}")
            return ""

    def main_ocr(self):

        st.title("OCR")

        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            # Read the uploaded image
            image = Image.open(uploaded_file)

            # Extract text
            extracted_text = Ocr.extract_text(image)

            # Display the uploaded image and extracted text
            st.image(image, caption="Uploaded Image")

            if extracted_text:
                st.success(f"Extracted Text: {extracted_text}")
            else:
                st.warning("No text found in the image.")


def main():

    st.title("PYTHON TOOLKIT")

    # Let the user choose which module to use
    selected_module = st.selectbox("", ["Ask your PDF 🤔💭", "FileQueryHub 📂🤖" ,"Optical Character Recognition"])

    if selected_module == "Ask your PDF 🤔💭":

        pdf =PDF()
        pdf.main_pdf()
    elif selected_module == "FileQueryHub 📂🤖":

        file=File()
        file.main_file()
    elif selected_module == "Optical Character Recognition":
        ocr=Ocr()
        ocr.main_ocr()






if __name__ == "__main__":
    main()

# from https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

import streamlit as st
from langchain_upstage import ChatUpstage as Chat

from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_upstage import UpstageDocumentParseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import tempfile, os

from PIL import Image
import math


def split_images(img_file_path):
    # It's image file with long and split them to right rectagle side with 50% overlap 
    
    # Open and process the image
    img = Image.open(img_file_path)
    width, height = img.size
    
    # Calculate dimensions for splitting
    section_height = width  # Make sections square
    overlap = section_height // 2
    
    # Calculate number of sections needed
    num_sections = math.ceil((height - overlap) / (section_height - overlap))
    
    # Create directory for split images
    split_dir = os.path.join(temp_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)
    
    # List to store all split image paths
    split_image_paths = []
    
    # Split the image into overlapping sections
    for i in range(num_sections):
        top = i * (section_height - overlap)
        bottom = min(top + section_height, height)
        
        # If this is the last section, adjust to include all remaining pixels
        if i == num_sections - 1:
            top = height - section_height
        
        section = img.crop((0, top, width, bottom))
        section_path = os.path.join(split_dir, f"section_{i}.png")
        section.save(section_path)
        split_image_paths.append(section_path)
    
    return split_image_paths

def img_to_doc_content(img_file_path):
    dp = UpstageDocumentParseLoader(img_file_path, split="page")
    docs = dp.load()  

    # get doc content
    doc_content = "\n".join([doc.page_content for doc in docs])
    return doc_content
    
def combine_doc_contents(all_doc_contents):
    basic_prompt = """You are processing text extracted from a long image that was split into overlapping sections. Your task is to:

1. Analyze multiple sections of text that have ~50% overlap with adjacent sections
2. Identify and remove redundant content from the overlapping areas
3. Maintain the correct sequence and flow of the text
4. Preserve all unique information
5. Ensure proper paragraph breaks and formatting
6. Return a single, coherent document that reads naturally
7. Please keep the original text and do not revise or translate it.

The following sections contain the extracted text, with overlapping content between them. Please combine them into one complete, non-redundant text while maintaining the original text and flow."""

    eval_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{basic_prompt}"),
            ("human", "{all_doc_contents}"),
        ]
    )   

    llm_chain = eval_prompt | llm | StrOutputParser()
    return llm_chain.stream({"basic_prompt": basic_prompt, "all_doc_contents": all_doc_contents})


st.title("Solar Long Image")

llm = Chat(model="solar-pro")


uploaded_file = st.file_uploader("Choose your long image file", type=["png", "jpg", "jpeg", "tiff", "bmp", "webp"])

if uploaded_file and uploaded_file.name:
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

            with st.status("Splitting the image..."):
                split_img_paths = split_images(file_path)
                st.write(f"Splitting the image done. Now we have {len(split_img_paths)} images.")

            all_doc_contents = []
            for i, path in enumerate(split_img_paths):
                with st.status(f"Processing the image [{i+1}/{len(split_img_paths)}]..."):
                    st.image(path)

                    doc_content = img_to_doc_content(path)
                    st.write(doc_content)

                    all_doc_contents.append(doc_content)

            with st.status("Combine all the contents...", expanded=True):
                combined_doc_content = combine_doc_contents(all_doc_contents)
                st.write_stream(combined_doc_content)


            # Let's remove all the split images
            for path in split_img_paths:
                os.remove(path)

            # It's image file with long and split them to right rectagle side with 50% overlap 


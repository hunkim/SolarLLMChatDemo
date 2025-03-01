import streamlit as st
from PIL import Image
import re
import io
import os
import qrcode
import base64
from openai import OpenAI
import json

# Set page title
st.set_page_config(page_title="Business Card OCR", layout="wide")
st.title("Business Card OCR to Contact")

# Function to extract text from image using Upstage API
def extract_info_from_image(image):
    try:
        # Convert PIL Image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Initialize Upstage API client
        client = OpenAI(
            api_key=st.secrets["UPSTAGE_API_KEY"],  # Store this in Streamlit secrets
            base_url="https://api.upstage.ai/v1/information-extract/"
        )
        
        # Call the API with the exact structure provided
        response = client.chat.completions.create(
            model="information-extraction",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            response_format={
              "type": "json_schema",
              "json_schema": {
                "name": "document_schema",
                "schema": {
                  "type": "object",
                  "properties": {
                    "companyName": {
                      "type": "string",
                      "description": "The name of the architecture company."
                    },
                    "address": {
                      "type": "string",
                      "description": "The physical address of the company."
                    },
                    "phone": {
                      "type": "string",
                      "description": "The telephone number of the company."
                    },
                    "mobile": {
                      "type": "string",
                      "description": "The mobile phone number of the contact person."
                    },
                    "email": {
                      "type": "string",
                      "description": "The email address of the contact person."
                    },
                    "fullname": {
                      "type": "string",
                      "description": "name of the person in biz card"
                    },
                    "title": {
                      "type": "string",
                      "description": "title of this person in biz card"
                    }
                  },
                  "required": [
                    "companyName",
                    "address",
                    "phone",
                    "mobile",
                    "email",
                    "fullname",
                    "title"
                  ]
                }
              }
            }
        )
        
        # Extract the structured data from the response
        extracted_info = json.loads(response.choices[0].message.content)
        return extracted_info
    except Exception as e:
        st.error(f"Error during information extraction: {e}")
        return None

def create_vcard(company_name, name, title, phone, mobile, email, address, website=""):
    """Create a vCard file from the extracted information"""
    vcard_content = f"""BEGIN:VCARD
VERSION:3.0
FN:{name}
ORG:{company_name}
TITLE:{title}
TEL;TYPE=WORK:{phone}
TEL;TYPE=CELL:{mobile}
EMAIL:{email}
ADR;TYPE=WORK:;;{address}
URL:{website}
END:VCARD
"""
    # Save vCard to file
    filename = f"{name.replace(' ', '_')}.vcf"
    with open(filename, "w") as f:
        f.write(vcard_content)
    return filename

def create_qr_code_for_vcard(vcard_file):
    """Create a QR code for the vCard file"""
    try:
        # Read vCard content
        with open(vcard_file, "r") as f:
            vcard_content = f.read()
        
        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(vcard_content)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Save QR code image
        qr_filename = vcard_file.replace(".vcf", "_qr.png")
        img.save(qr_filename)
        return qr_filename
    except Exception as e:
        st.error(f"Error generating QR code: {e}")
        return None

# --- Main App Section ---

# Add API key input in sidebar
if "UPSTAGE_API_KEY" not in st.secrets:
    st.sidebar.title("API Configuration")
    api_key = st.sidebar.text_input("Enter Upstage API Key", type="password")
    if api_key:
        st.secrets["UPSTAGE_API_KEY"] = api_key
    else:
        st.warning("Please enter your Upstage API key in the sidebar to continue.")

st.header("Upload a Business Card Image")

# Create a radio button for selecting input method
input_method = st.radio("Choose input method:", ["Upload Image", "Take Photo"], horizontal=True, label_visibility="collapsed")

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="uploader")
    camera_file = None
else:  # Take Photo
    camera_file = st.camera_input("Capture business card", key="camera")
    uploaded_file = None

# Determine which input method was used
input_file = uploaded_file or camera_file

if input_file is not None and "UPSTAGE_API_KEY" in st.secrets:
    # Display the uploaded image
    image = Image.open(input_file)
    img_col, info_col = st.columns(2)
    
    with img_col:
        st.image(image, caption="Uploaded Business Card", use_container_width=True)
    
    # Extract information using Upstage API
    with st.spinner("Extracting information from business card..."):
        extracted_info = extract_info_from_image(image)
    
    if extracted_info:
        with info_col:
            st.subheader("Extracted Information")
            st.json(extracted_info)
        
        # Create form for editing extracted information
        st.subheader("Edit Contact Information")
        
        # Auto-generate contact initially
        if "contact_generated" not in st.session_state:
            name = extracted_info.get("fullname", "")
            title = extracted_info.get("title", "")
            company_name = extracted_info.get("companyName", "")
            address = extracted_info.get("address", "")
            phone = extracted_info.get("phone", "")
            mobile = extracted_info.get("mobile", "")
            email = extracted_info.get("email", "")
            website = ""
            
            if name:
                # Create vCard
                vcard_file = create_vcard(company_name, name, title, phone, mobile, email, address, website)
                
                # Create QR code
                qr_file = create_qr_code_for_vcard(vcard_file)
                
                st.session_state.contact_generated = True
                st.session_state.vcard_file = vcard_file
                st.session_state.qr_file = qr_file
        
        with st.form("contact_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Full Name", extracted_info.get("fullname", ""))
                title = st.text_input("Job Title", extracted_info.get("title", ""))
                company_name = st.text_input("Company Name", extracted_info.get("companyName", ""))
                address = st.text_input("Address", extracted_info.get("address", ""))
            
            with col2:
                phone = st.text_input("Phone Number", extracted_info.get("phone", ""))
                mobile = st.text_input("Mobile Number", extracted_info.get("mobile", ""))
                email = st.text_input("Email Address", extracted_info.get("email", ""))
                website = st.text_input("Website/Social Media", "")
            
            regenerate_button = st.form_submit_button("Regenerate Contact")
        
        if regenerate_button:
            if not name:
                st.error("Please provide at least a name.")
            else:
                # Create vCard
                vcard_file = create_vcard(company_name, name, title, phone, mobile, email, address, website)
                
                # Create QR code
                qr_file = create_qr_code_for_vcard(vcard_file)
                
                st.session_state.vcard_file = vcard_file
                st.session_state.qr_file = qr_file
        
        # Display contact information if available
        if "contact_generated" in st.session_state:
            vcard_file = st.session_state.get("vcard_file")
            qr_file = st.session_state.get("qr_file")
            
            # Display success message and QR code
            st.success(f"Contact information saved to {vcard_file}")
            
            if qr_file and os.path.exists(qr_file):
                qr_image = Image.open(qr_file)
                st.image(qr_image, caption="Scan this QR code to add contact", width=300)
            
            # Provide download buttons
            if vcard_file and os.path.exists(vcard_file):
                with open(vcard_file, "rb") as file:
                    vcf_contents = file.read()
                    st.download_button(
                        label="Download vCard File",
                        data=vcf_contents,
                        file_name=vcard_file,
                        mime="text/vcard"
                    )
            
            if qr_file and os.path.exists(qr_file):
                with open(qr_file, "rb") as file:
                    qr_contents = file.read()
                    st.download_button(
                        label="Download QR Code",
                        data=qr_contents,
                        file_name=qr_file,
                        mime="image/png"
                    )
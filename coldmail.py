# from https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

import streamlit as st
from langchain_upstage import ChatUpstage as Chat

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

from solar_util import initialize_solar_llm

from solar_util import prompt_engineering

import json
from pathlib import Path

llm = initialize_solar_llm()
st.set_page_config(page_title="Cold Email Generator", layout="wide")
st.title("B2B Cold Email Generator")


chat_with_history_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", """You are Solar, a smart chatbot by Upstage, loved by many people. 
         Be smart, cheerful, and fun. Give engaging answers and avoid inappropriate language.
         reply in the same language of the user query.
         Solar is now being connected with a human.
         
         Please put <END> in the end of your answer."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{user_query}"),
    ]
)



def get_response(user_query, chat_history):
    chain = chat_with_history_prompt | llm | StrOutputParser()

    response = ""
    end_token = ""
    for chunk in chain.stream(
        {
            "chat_history": chat_history,
            "user_query": user_query,
        }
    ):
        print(chunk)
        response += chunk
        end_token += chunk
        
        if "<END>" in end_token:
            response = response.split("<END>")[0]
            break
        
        # Keep only the last 5 characters to check for <END>
        end_token = end_token[-5:]
        
        yield chunk

    yield response


# Add these constants for our cold email structure
COLD_EMAIL_TEMPLATE = """You are a professional cold email writer.
Based on the following information, create a compelling cold email:
- Your Company: {company_name}
- Your Product/Service: {product}
- Your Contact Information: {contact_info}

Target Company Information:
- Company Name: {target_company_name}
- Business Description: {target_business}
- Contact Email: {target_email}

Use the example email as a reference for tone and structure:
{example_email}

Make the email professional, concise, and persuasive.
Include a clear value proposition and call to action that's specific to the target company's business.
reply in the same language of the user query.

Please put <END> in the end of your answer."""

# Add these to track cold email information
if "cold_email_info" not in st.session_state:
    st.session_state.cold_email_info = {
        "company_name": "Upstage.AI",
        "product": "We specialize in Document AI and Large Language Models (LLMs), offering cutting-edge solutions that combine both technologies. Our products help businesses automate document processing, enhance information extraction, and leverage advanced AI capabilities for improved efficiency and decision-making.",
        "contact_info": "contact@upstage.ai",
        "target_companies": "Enterprise companies seeking advanced AI solutions for document processing and natural language understanding",
        "cold_email_example": """Subject: Enhancing Coupang's E-commerce Experience with AI Solutions

Dear Coupang Team,

I hope this email finds you well. I am reaching out from Upstage.AI, a leading provider of Document AI and Large Language Model solutions, as I believe we could add significant value to Coupang's e-commerce operations.

Given Coupang's position as South Korea's largest e-commerce platform, I wanted to explore how our AI solutions could enhance your shopping experience. Our technology can help:

• Improve product search accuracy and recommendations
• Automate product description processing and categorization
• Enhance customer service through advanced AI chatbots
• Streamline document processing for vendor onboarding

Would you be open to a brief conversation about how these solutions could benefit Coupang's operations?

Please feel free to reach out to us at contact@upstage.ai to schedule a discussion.

Best regards,
Upstage.AI Team""",
        "additional_notes": ""
    }

def load_target_companies():
    json_path = Path(__file__).parent / "data" / "target_companies.json"
    with open(json_path, 'r') as f:
        return json.load(f)['target_companies']

def generate_emails(company_info):
    target_companies = load_target_companies()
    emails = []
    
    st.markdown("## Generating Cold Emails")
    
    for idx, target in enumerate(target_companies, 1):
        company_container = st.container()
        with company_container:
            st.markdown(f"### 📧 {target['company_name']} ({idx}/{len(target_companies)})")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("🎯 **Target Company Information**")
                st.markdown(f"""
                - Company Name: {target['company_name']}
                - Main Business: {target['main_business']}
                - Contact Email: {target['contact_email']}
                """)
            
            chain = ChatPromptTemplate.from_messages([
                ("human", COLD_EMAIL_TEMPLATE)
            ]) | llm | StrOutputParser()
            
            try:
                with col2:
                    st.write("⚙️ Generating personalized content...")
                
                response = chain.invoke({
                    "company_name": company_info["company_name"],
                    "product": company_info["product"],
                    "contact_info": company_info["contact_info"],
                    "target_company_name": target["company_name"],
                    "target_business": target["main_business"],
                    "target_email": target["contact_email"],
                    "example_email": company_info["cold_email_example"]
                })
                
                email_content = response.split("<END>")[0].strip()
                emails.append({
                    "target_company": target["company_name"],
                    "email_content": email_content,
                    "status": "success"
                })
                
                with col2:
                    st.write("✅ Email generated successfully!")
                    st.markdown("#### Generated Email")
                    st.markdown(f"""
                        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 5px;">
                            {email_content}
                        </div>
                        """, unsafe_allow_html=True)
                    st.button(
                        "📋 Copy to Clipboard",
                        key=f"copy_{target['company_name']}",
                        on_click=lambda text=email_content: st.write(text)
                    )
            
            except Exception as e:
                emails.append({
                    "target_company": target["company_name"],
                    "email_content": f"Error generating email: {str(e)}",
                    "status": "error"
                })
                with col2:
                    st.write("❌ Error occurred during generation")
                    st.error(f"Error: {str(e)}")
            
            st.divider()  # Add visual separation between emails
    
    # Show summary statistics at the end
    st.markdown("## Summary")
    col1, col2, col3 = st.columns(3)
    
    total_emails = len(emails)
    successful_emails = sum(1 for email in emails if email["status"] == "success")
    failed_emails = total_emails - successful_emails
    
    col1.metric("Total Emails", total_emails)
    col2.metric("Successful", successful_emails)
    col3.metric("Failed", failed_emails)
    
    return emails

# Remove the sidebar wrapper and organize content in the main area
st.subheader("Email Generator Settings")

st.session_state.cold_email_info["company_name"] = st.text_input(
    "Your Company Name", 
    st.session_state.cold_email_info["company_name"]
)
st.session_state.cold_email_info["product"] = st.text_area(
    "Product/Service Description", 
    st.session_state.cold_email_info["product"],
    height=100
)
st.session_state.cold_email_info["contact_info"] = st.text_input(
    "Contact Information", 
    st.session_state.cold_email_info["contact_info"]
)
st.session_state.cold_email_info["additional_notes"] = st.text_area(
    "Example Email", 
    st.session_state.cold_email_info["cold_email_example"],
    height=300
)

# Generate button
if st.button("Generate Cold Email", type="primary"):
    if not st.session_state.cold_email_info["company_name"]:
        st.error("Please enter your company name")
    else:
        generated_emails = generate_emails(st.session_state.cold_email_info)
        
        # Display generated emails in the main area with better formatting
        st.markdown("## Generated Cold Emails")
        
        # Create three columns for statistics
        col1, col2, col3 = st.columns(3)
        
        # Calculate statistics
        total_emails = len(generated_emails)
        successful_emails = sum(1 for email in generated_emails if email["status"] == "success")
        failed_emails = total_emails - successful_emails
        
        # Display statistics in metrics
        col1.metric("Total Emails", total_emails)
        col2.metric("Successful", successful_emails)
        col3.metric("Failed", failed_emails)
        
        # Display emails with better formatting
        for email in generated_emails:
            with st.expander(f"📧 {email['target_company']}", expanded=False):
                if email["status"] == "success":
                    st.markdown("### Email Content")
                    st.markdown(f"""
                        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 5px;">
                            {email["email_content"]}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Add copy button
                    st.button(
                        "📋 Copy to Clipboard",
                        key=f"copy_{email['target_company']}",
                        on_click=lambda text=email["email_content"]: st.write(text)
                    )
                else:
                    st.error(email["email_content"])

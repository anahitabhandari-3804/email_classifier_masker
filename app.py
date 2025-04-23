import streamlit as st
import requests
import json

# Set Streamlit page config
st.set_page_config(page_title="Email Classifier & PII Masking", layout="centered")

st.title("📧 Email Classification & PII Masking App")
st.markdown("Enter an email body below to detect PII, classify the category, and get a masked version.")

email_body = st.text_area("Email Body", height=300, placeholder="Type or paste your email here...")

if st.button("Submit"):
    if not email_body.strip():
        st.warning("Please enter a valid email body.")
    else:
        with st.spinner("Analyzing..."):
            try:
                # URL for the locally hosted FastAPI server
                api_url = "http://localhost:8000/classify"
                payload = {"email_text": email_body}
                headers = {"Content-Type": "application/json"}

                # Make the API request
                response = requests.post(api_url, data=json.dumps(payload), headers=headers)

                # Check if the request was successful
                if response.status_code == 200:
                    result = response.json()
                    st.subheader("📄 Masked Email")
                    st.code(result["masked_email"], language="text")
                    st.subheader("🏷️ Category")
                    st.success(f"**{result['category_of_the_email']}**")
                    st.subheader("🔐 Masked Entities")
                    if result["list_of_masked_entities"]:
                        for entity in result["list_of_masked_entities"]:
                            start, end = entity["position"]
                            st.markdown(f"- **{entity['classification']}** at *{start}-{end}*: `{entity['entity']}`")
                    else:
                        st.info("No PII entities detected.")
                else:
                    st.error(f"Error from API: {response.status_code} - {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the FastAPI server. Ensure it is running at http://localhost:8000.")
            except Exception as e:
                st.exception(f"An error occurred: {e}")
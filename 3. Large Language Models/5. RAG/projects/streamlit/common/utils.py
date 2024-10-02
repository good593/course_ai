import os
import base64

def display_pdf(streamlit, file):
    # Opening file from file path

    streamlit.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""

    # Displaying File
    streamlit.markdown(pdf_display, unsafe_allow_html=True)

def upload_file(streamlit, temp_dir, uploaded_file):
    file_path = os.path.join(temp_dir, uploaded_file.name)
                
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    streamlit.write("Indexing your document...")

    if not os.path.exists(file_path):
        streamlit.error('Could not find the file you uploaded, please check again...')
        streamlit.stop()
    
    file_key = f"{streamlit.session_state.id}-{uploaded_file.name}"
    return file_key, file_path
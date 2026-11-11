import streamlit as st
import base64
from datetime import datetime
import backend

def main():
    st.set_page_config(page_title="PDF Chat", page_icon="📚")
    st.title("RAG Document Search 🔎")
    st.caption("Ask questions about the loaded documents")

    # Initialise session state
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "indexed_pdf_names" not in st.session_state:
        st.session_state.indexed_pdf_names = []
    if "last_processed_files" not in st.session_state:
        st.session_state.last_processed_files = []
    
    # Handle delayed clearing of question input
    if st.session_state.get("clear_input"):
        st.session_state.question_input = ""
        st.session_state.clear_input = False

    # ── sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        # st.title("📚 PDF Chat")

        if not backend.API_KEY:
            st.error("GOOGLE_API_KEY not found in .env file.")
            st.stop()
        else:
            st.success("API key loaded ✓", icon="🔑")

        pdf_docs = st.file_uploader(
            "Upload PDF files", accept_multiple_files=True, type=["pdf"]
        )

        current_file_names = [p.name for p in pdf_docs] if pdf_docs else []
        
        if st.button("⚙️ Process PDFs", use_container_width=True):
            if not pdf_docs:
                st.warning("Please upload at least one PDF first.")
            elif current_file_names == st.session_state.last_processed_files:
                st.info("Files are already processed.")
            else:
                print(f"Frontend: Processing {len(pdf_docs)} files...")
                with st.spinner("Indexing PDFs..."):
                    vector_store, _ = backend.build_vector_store(pdf_docs)
                    st.session_state.vector_store = vector_store
                    st.session_state.indexed_pdf_names = current_file_names
                    st.session_state.last_processed_files = current_file_names
                st.success(f"Indexed {len(pdf_docs)} PDF(s) ✓")
                print("Frontend: PDF processing complete.")

        if st.session_state.indexed_pdf_names:
            st.caption("📄 Indexed: " + ", ".join(st.session_state.indexed_pdf_names))

        st.divider()

        col1, col2 = st.columns(2)
        if col1.button("🔄 Reset all", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.vector_store = None
            st.session_state.indexed_pdf_names = []
            st.session_state.last_processed_files = []
            st.rerun()
        if col2.button("🗑 Clear last", use_container_width=True):
            if st.session_state.conversation_history:
                st.session_state.conversation_history.pop()
                st.rerun()

        # TXT download
        if st.session_state.conversation_history:
            history_text = ""
            for entry in st.session_state.conversation_history:
                history_text += f"Question: {entry['Question']}\n"
                history_text += f"Answer: {entry['Answer']}\n"
                history_text += f"Timestamp: {entry['Timestamp']}\n"
                history_text += f"PDFs: {entry['PDF Name']}\n"
                history_text += "-" * 50 + "\n\n"
            
            b64 = base64.b64encode(history_text.encode()).decode()
            st.markdown(
                f'<a href="data:file/txt;base64,{b64}" download="conversation_history.txt">'
                f'<button style="width:100%;margin-top:1rem;cursor:pointer;">⬇ Download history (.txt)</button></a>',
                unsafe_allow_html=True,
            )

    # ── main area ─────────────────────────────────────────────────────────────
    # Display chat history container
    history_container = st.container()

    # User input at the bottom
    st.divider()
    with st.form("ask_form", border=False):
        col_q, col_btn = st.columns([5, 1])
        with col_q:
            user_question = st.text_input(
                "Your question",
                placeholder="What would you like to know?",
                label_visibility="collapsed",
                key="question_input",
            )
        with col_btn:
            ask_clicked = st.form_submit_button("Ask ➤", use_container_width=True)

    if ask_clicked:
        if not user_question.strip():
            st.warning("Please type a question first.")
        elif st.session_state.vector_store is None:
            st.warning("Please upload and process your PDFs first.")
        else:
            print(f"Frontend: Handling question: '{user_question.strip()}'")
            # Build history string for continuity
            history_str = ""
            for entry in st.session_state.conversation_history[-5:]: # Last 5 for context
                history_str += f"User: {entry['Question']}\nAssistant: {entry['Answer']}\n"

            with st.spinner("Thinking..."):
                #asking query
                answer = backend.get_answer(
                    user_question.strip(), 
                    st.session_state.vector_store, 
                    history_str.strip(),
                    st.session_state.indexed_pdf_names
                )
            
            entry = {
                "Question": user_question.strip(),
                "Answer": answer,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "PDF Name": ", ".join(st.session_state.indexed_pdf_names),
            }
            st.session_state.conversation_history.append(entry)
            print("Frontend: Question answered and added to history.")
            st.session_state.clear_input = True
            st.rerun()

    # Render History in the container (oldest at top, newest at bottom)
    with history_container:
        # if not st.session_state.conversation_history:
        #     st.info("No conversation yet. Upload PDFs and ask a question!")
        for entry in st.session_state.conversation_history:
            with st.chat_message("user"):
                st.markdown(entry['Question'])
            
            with st.chat_message("assistant"):
                st.markdown(entry['Answer'])
                st.caption(f"🕒 {entry['Timestamp']}  |  📄 {entry['PDF Name']}")

if __name__ == "__main__":
    main()
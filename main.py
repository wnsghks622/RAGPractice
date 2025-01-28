import streamlit as st

# session states
## chat history with initial message
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "bot", "content": "Hello, how can I help you?"}]
    
## Store current query (i.e., last user input)
if "current_query" not in st.session_state:
    st.session_state.current_query = ""

def run_chatui():
    st.set_page_config(page_title="🤗💬 RAG Bot")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    # We instantiate a new prompt with each chat input because streamlit reruns everything
    if prompt := st.chat_input():
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            st.session_state.current_query = prompt

    # Generate a new response if last message is not from bot
    if st.session_state.chat_history[-1]["role"] != "bot":
        with st.chat_message("RAG bot"):
            with st.spinner("Thinking..."):
                # We'll have it repeat us for now, like a baby bot. 
                # Here is where the LLM response would go. 
                answer = st.session_state.current_query
                st.write("🤖: " + answer)
            
        message = {"role": "bot", "content": answer}
        st.session_state.chat_history.append(message)
    
if __name__ == "__main__":
    run_chatui()
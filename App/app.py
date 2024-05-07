


import streamlit as st
from streamlit.external.langchain import StreamlitCallbackHandler

from docProcessing import configure_retrieval_chain, MEMORY


st.set_page_config(page_title="Sakura Sky", page_icon="")
st.title("Sakura Sky")




if st.sidebar.button("Clear message history"):
    MEMORY.chat_memory.clear()

avatars = {"human": "user", "ai": "assistant"}

if len(MEMORY.chat_memory.messages) == 0:
    st.chat_message("assistant").markdown("Ask me anything!")

for msg in MEMORY.chat_memory.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

assistant = st.chat_message("assistant")
if user_query := st.chat_input(placeholder="Give me 3 keywords for what you have right now"):
    st.chat_message("user").write(user_query)
    container = st.empty()
    stream_handler = StreamlitCallbackHandler(container)
    with st.chat_message("assistant"):
        if use_flare:
            params = {
                "user_input": user_query,
            }
        else:
            params = {
                "question": user_query,
                "chat_history": MEMORY.chat_memory.messages,
            }
        response = CONV_CHAIN.run(params, callbacks=[stream_handler])
        # Display the response from the chatbot
        if response:
            container.markdown(response)

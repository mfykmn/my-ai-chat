import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def main():
    st.set_page_config(
        page_title="My AI Chat",
        page_icon=":robot:",
    )
    st.header("My AI Chat :robot:")
    
    # チャット履歴の初期化
    if "message_history" not in st.session_state:
        st.session_state.message_history = [
            # System Promptを設定
            ("system", "あなたは優秀なアシスタントです。")
        ]
    
    llm = ChatOpenAI(
        temperature=0,
        model="deepseek-chat",
        openai_api_key=os.getenv("DEEP_SEEK_API_KEY"),
        openai_api_base="https://api.deepseek.com",
    )
    
    prompt = ChatPromptTemplate.from_messages([
        *st.session_state.message_history,
        ("user", "{user_input}")
    ])
    
    output_parser = StrOutputParser()
    
    chain = prompt | llm | output_parser
    
    if user_input := st.chat_input("聞きたいことを入力してください"):
        with st.spinner("AI is typing ..."):
            response = chain.invoke({"user_input": user_input})
            
        # チャット履歴にユーザーの入力を追加
        st.session_state.message_history.append(("user", user_input))
        # チャット履歴にAIの応答を追加
        st.session_state.message_history.append(("ai", response))
    
    for role, message in st.session_state.get("message_history", []):
        st.chat_message(role).markdown(message)

if __name__ == '__main__':
    main()        
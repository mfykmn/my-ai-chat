# my-ai-chat

## 環境構築
```
conda create -n my_ai_chat_env python=3.12
conda activate my_ai_chat_env
conda env list
```

## 動作確認
```
$ pip install -r requirements.txt
$ streamlit run main.py
```

## 学習メモ
### temperature
temperatureが高い(ex 0.8, 0.9)場合、より多様な回答が出るが、その分関連性の低い回答を返したりハルシネーションが起きやすい。
temperatureが低い(ex 0.2, 0.1)場合、その分関連性の高い回答を返すが、その分多様性がなくなる。
### LangChainのMemory Component
LLMとのやり取りの履歴を保持するコンポーネント

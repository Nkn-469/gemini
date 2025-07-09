import os
import sys
import google.generativeai as genai
from dotenv import load_dotenv
import argparse

def main():
    """Gemini APIと対話的なチャットを行うメイン関数"""
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ エラー: 環境変数 'GOOGLE_API_KEY' が設定されていません。", file=sys.stderr)
        print("💡 ヒント: .envファイルに GOOGLE_API_KEY='あなたのキー' を追加してください。", file=sys.stderr)
        sys.exit(1)

    genai.configure(api_key=api_key)

    try:
        model = genai.GenerativeModel('gemini-1.0-pro')
        # 会話履歴を保持するチャットセッションを開始
        chat = model.start_chat(history=[])

        print("🤖 Geminiとの対話を開始します。(終了するには 'quit' または 'exit' と入力してください)")
        print("-" * 60)

        while True:
            prompt = input("You: ")

            if prompt.lower() in ["quit", "exit"]:
                print("\n👋 対話を終了します。")
                break

            if not prompt.strip():
                continue

            print("Gemini: ", end="", flush=True)
            response = chat.send_message(prompt, stream=True)
            for chunk in response:
                print(chunk.text, end="", flush=True)
            print() # 改行

    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

質問後は空行を入力するまで続けて質問でき、会話履歴が引き継がれる。
ツールは持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import time

from google import genai
from google.genai.chats import Chat
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, chat: Chat | None = None) -> str:
    if chat is None:
        client = genai.Client(api_key=get_api_key())
        chat = client.chats.create(model=MODEL)
    response = chat.send_message(question)

    usage = response.usage_metadata
    if usage is not None:
        token_counts = [
            ("入力", usage.prompt_token_count),
            ("出力", usage.candidates_token_count),
            ("思考", usage.thoughts_token_count),
            ("合計", usage.total_token_count),
        ]
        available_counts = [f"{label}: {count}" for label, count in token_counts if count is not None]
        if available_counts:
            print(f"[トークン数] {'、'.join(available_counts)}")

    return response.text or ""


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        client = genai.Client(api_key=get_api_key())
        chat = client.chats.create(model=MODEL)
        question = args.question
        while True:
            print(f"[質問] {question}\n")
            started_at = time.perf_counter()
            print(ask(question, chat))
            print(f"[回答時間] {time.perf_counter() - started_at:.2f} 秒")
            question = input("\n続けて質問してください（終了するには空行）: ")
            if not question.strip():
                break
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)
    except EOFError:
        pass


if __name__ == "__main__":
    main()

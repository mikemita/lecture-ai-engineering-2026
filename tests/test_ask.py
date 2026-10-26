from types import SimpleNamespace

from app import ask as ask_module


def test_main_keeps_chat_history_for_follow_up_questions(monkeypatch, capsys):
    class FakeChat:
        def __init__(self):
            self.questions = []

        def send_message(self, question):
            self.questions.append(question)
            return SimpleNamespace(
                text=f"answer to {question}",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=12,
                    candidates_token_count=5,
                    thoughts_token_count=2,
                    total_token_count=19,
                ),
            )

    chat = FakeChat()

    class FakeClient:
        chats = SimpleNamespace(create=lambda *, model: chat)

    monkeypatch.setattr(ask_module.genai, "Client", lambda *, api_key: FakeClient())
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    timestamps = iter([10.0, 11.25, 20.0, 22.5])
    monkeypatch.setattr(ask_module.time, "perf_counter", lambda: next(timestamps))
    monkeypatch.setattr("sys.argv", ["app.ask", "最初の質問"])
    questions = iter(["続けて質問", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(questions))

    ask_module.main()

    assert chat.questions == ["最初の質問", "続けて質問"]
    output = capsys.readouterr().out
    assert "[トークン数] 入力: 12、出力: 5、思考: 2、合計: 19" in output
    assert output.index("answer to 最初の質問") < output.index("[回答時間] 1.25 秒")
    assert "[回答時間] 2.50 秒" in output

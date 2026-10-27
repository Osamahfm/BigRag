from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from config import ROOT


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_ui_examples_chat_followup_reset_and_evaluation(tmp_path):
    with patch("app.data_directory", return_value=tmp_path), patch.dict("os.environ", {"RAG_DATA_DIR": str(tmp_path), "GEMINI_API_KEY": ""}):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        assert not app.exception and app.chat_input[0].disabled
        button(app, "Load example documents").click().run()
        assert not app.exception and not app.chat_input[0].disabled
        assert app.metric[0].value == "3"
        app.chat_input[0].set_value("What is MapReduce?").run()
        assert not app.exception and len(app.chat_message) == 2
        app.chat_input[0].set_value("What are its advantages?").run()
        assert not app.exception and len(app.chat_message) == 4
        app.radio(key="navigation").set_value("Evaluation").run()
        button(app, "Run comparison").click().run()
        assert not app.exception and len(app.dataframe) >= 1
        app.radio(key="navigation").set_value("Workspace").run()
        button(app, "New conversation").click().run()
        assert not app.exception and len(app.chat_message) == 0
        app.chat_input[0].set_value("ما مزايا MapReduce؟").run()
        assert not app.exception and len(app.chat_message) == 2
        button(app, "Clear conversation").click().run()
        assert not app.exception and len(app.chat_message) == 0


def test_ui_empty_document_selection_and_missing_key(tmp_path):
    with patch.dict("os.environ", {"RAG_DATA_DIR": str(tmp_path), "GEMINI_API_KEY": ""}):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        button(app, "Load example documents").click().run()
        app.multiselect[0].set_value([]).run()
        assert not app.exception and app.chat_input[0].disabled
        app.selectbox(key="backend").set_value("Gemini").run()
        assert not app.exception and app.chat_input[0].disabled
        assert any("missing" in w.value for w in app.warning)

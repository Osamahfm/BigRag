"""Atlas entry point: python -m streamlit run app.py."""
from config import data_directory
from ui.application import main

if __name__ == "__main__":
    main(data_directory())

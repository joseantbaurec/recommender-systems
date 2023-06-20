import streamlit as st
from annotated_text import annotation

HIGHLIGHT_BLUE = '#5c99fa'
HIGHLIGHT_RED = '#fa4646'
HIGHLIGHT_GRAY = '#98989e'
HIGHLIGHT_GREEN = '#71c967'


def setup_page():
    st.set_page_config(
        page_title='Sistemas de Recomendación',
        page_icon='res/icons/favicon.ico',
        layout='wide',
    )
    st.markdown(
        """
    <style>
    h1, h2 {
        font-family: 'Futura', sans-serif;
    }
    h1 {
        font-size: 4em
    }
    p, li {
        margin: 10px;
        font-family: 'Source Code Pro', monospace;
        font-size: 14px;
    }
    .stApp {
        background: rgb(255,255,255);
        background: linear-gradient(149deg, rgba(255,255,255,1) 0%, rgba(255,175,0,1) 53%, rgba(85,87,107,1) 97%);
    }
    .css-18ni7ap {
        visibility: visible;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def vspace(n_lines: int):
    for n in range(n_lines):
        st.markdown("")


def highlight(text: str, background_color: str) -> str:
    return str(annotation(text, background=background_color))


def write(text: str):
    st.write(text, unsafe_allow_html=True)


def bullet_item(text: str):
    write('▶️ ' + text)

import pickle
import random

import streamlit as st

import utils as colors
from base.dataset import DataSet
from models.word2vec_model import Word2VecRecommender
from utils import bullet_item, highlight, setup_page, vspace, write

# page configs
setup_page()

# model loading
@st.cache_data(show_spinner='¡Cargando modelo! Espera, por favor...')
def load_word2vec(path: str) -> Word2VecRecommender:
    with open(path, 'rb') as file:
        w2v: Word2VecRecommender = pickle.load(file)
        return w2v


# page content
st.title('Pruébalo tu mismo')
vspace(2)
col1, _ = st.columns([7, 4])
with col1:
    write(
        'En esta página podrás interactuar en vivo con el modelo '
        + highlight('**Word2Vec**', colors.HIGHLIGHT_GREEN)
        + ':'
    )
    bullet_item('Primero, se elige un <u>*usuario*</u> al azar.')
    bullet_item(
        'Obtenemos la lista de todos los productos que ese usuario considera **relevantes**, y elegimos uno al azar.'
    )
    bullet_item(
        'Usando ese producto como *prompt* para el modelo, sacamos las 15 primeras recomendaciones que ofrece **Word2Vec**.'
    )

    vspace(1)
    user_bt = st.button('⚙️ Generar recomendación')
    if user_bt:
        model: Word2VecRecommender = load_word2vec(
            '/Users/josean/Desktop/Playground/recommender-systems/data/models/word2vec_model.pickle'
        )
        dataset: DataSet = model.dataset
        vspace(1)
        st.header('👥 Tu usuario')
        user = dataset.get_random_user()
        relevant_items = dataset.users.loc[user.name, 'relevant_items']
        relevant_items = dataset.products.loc[
            relevant_items, ['category_code', 'brand', 'price']
        ]
        selected_item = random.randint(0, len(relevant_items))
        selected_item = relevant_items.index[selected_item]

        def table_style(row, anchor):
            if row.name == anchor:
                style = ['background-color: lightgray']
            else:
                style = ['']
            return style * len(row)

        write(
            f'Tu usuario es <u>*{user.name}*</u>, y estos son los productos que considera relevantes:'
        )
        table = st.dataframe(
            relevant_items.style.apply(table_style, anchor=selected_item, axis=1),
            use_container_width=True,
        )
        write(
            f'El producto elegido como *prompt* para el modelo es <u>*{selected_item}*</u>'
        )
        vspace(2)
        st.header('🧩️ Las recomendaciones')
        vspace(1)
        res, output = model.recommend(
            user.name, selected_item, n_recommendations=15, silent=True
        )
        matches = res.matches
        matches.index = output.index
        output['¿Acertada?'] = res.matches
        write('Aquí puedes ver las 15 primeras recomendaciones que nos da el modelo:')
        recs = st.dataframe(output, use_container_width=True)

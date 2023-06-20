import pandas as pd
import streamlit as st

import utils as colors
from base.plotting import plotly_from_json
from utils import bullet_item, highlight, setup_page, vspace, write

# page configs
setup_page()

# page content
st.title('Modelos basados en NLP')
vspace(3)
col1, _ = st.columns([8.5, 1])
with col1:
    write(
        'Para definir '
        + highlight('**similitud**', colors.HIGHLIGHT_GRAY)
        + ' podemos acercarnos al mundo del <u>*Natural Language Processing*</u> y obtener inspiración. Como hicimos en el modelo de *co-popularidad*, suponemos que '
        'dos productos son **similares** si reciben interacciones por un mismo usuario. Sin embargo, en vez de hacer un simple conteo, lo miramos desde el punto de vista '
        'de la *frecuencia en contexto*.'
    )
    write(
        'Supongamos que los usuarios interactúan con los productos en '
        + highlight('**sesiones**', colors.HIGHLIGHT_BLUE)
        + ', franjas de tiempo en las que todas las interacciones que realiza un mismo usuario son *consecutivas*, o comparten algún tipo de *contexto común*. Por ejemplo, '
        'un usuario puede ver seguidos dos capitulos de una misma *sitcom*, y al día siguiente ver tres documentales sobre ballenas; no quiere decir que los documentales de '
        'ballenas sean similares a las *sitcoms*.'
    )
    write(
        'Los modelos clásicos en NLP de <u>*word embeddings*</u> son muy útiles en estos casos: los *productos* se convierten en '
        + highlight('**palabras**', colors.HIGHLIGHT_GRAY)
        + ', mientras que las *sesiones* se convierten en '
        + highlight('**frases**', colors.HIGHLIGHT_GRAY)
        + '. Así, estudiar la frecuencia y orden de los productos dentro de las sesiones nos proporciona un '
        + highlight('**embedding**', colors.HIGHLIGHT_GREEN)
        + ' del banco de productos en un espacio de dimensión "pequeña", donde las palabras están más cerca cuanto más frecuentemente aparecen juntas en una sesión.'
    )

vspace(3)
col1, _, col2, _ = st.columns([6, 0.5, 4.5, 1])
with col1:
    st.header('💬 Recomendador Word2Vec 📐')
    write(
        'Uno de los algoritmos más clásicos de *word embeddings* es **Word2Vec**. Se trata de una red neuronal con una sola capa densa oculta (el *embedding*), y tanto las capas de '
        'input como de output representan el *one-hot encoding* del conjunto de palabras.'
    )
    write(
        'Dos modalidades existen para este modelo, **Skip-Gram** y **Continous Bag of Words**: *Skip-Gram* nos da la probabilidad de que cada palabra aparezca "cerca" de una palabra dada, '
        'mientras que *CBoW* nos da la probabilidad de que cada palabra aparezca "en medio" de una frase dada. Ambas pueden ser útiles, pero la primera modalidad suele ser la más utilizada.'
    )
    write(
        'Gracias a la implementación, podemos hacer **consultas agregadas**: buscar a partir del historial de un usuario, en vez de utilizar un único producto como *prompt* para el modelo.'
    )
with col2:
    vspace(5)
    st.image('res/images/word2vec.png')
vspace(2)
col1, _ = st.columns([8.5, 1])
with col1:
    write(
        'Aunque es difícil de navegar, podemos visualizar y explorar el *embedding* que aprende Word2Vec en tres dimensiones a través de la siguiente gráfica:'
    )
    fig = plotly_from_json('res/figures/word2vec_embedding.json')
    fig.update_layout(width=1225, height=750)
    fig.update_traces(marker_line_width=0, opacity=0.5)
    st.plotly_chart(fig)
col1, _ = st.columns([8.5, 1])
with col1:
    st.header('🧮 Estadísticas 💯')
    vspace(1)
    write('Evaluemos nuestro modelo con las mismas métricas base anteriores:')
    vspace(1)

_, col1, _, col2, _ = st.columns([0.1, 3, 0.1, 3, 0.7])
with col1:
    write('<u>*Entrenamiento*</u>')
    metrics = [
        ['1h 40min', '-', '-', '-'],
        ['0.002s', '0.002s', '0.035s', '0.002s'],
        ['0.0591', '0.0200', '0.5911', '0'],
        ['0.0752', '0.0445', '0.4545', '0'],
        ['12.10%', '10%', '70%', '0%'],
        ['59.00%', '100%', '100%', '0%'],
        ['3.7458', '3', '10', '-'],
        ['0.2597', '0.1429', '1', '0'],
    ]
    cols = ['Media', 'Mediana', 'Más alto', 'Más bajo']
    indx = [
        'T. Entrenamiento',
        'T. Evaluación',
        'MAP@k',
        'R@k',
        'P@k',
        'HR@k',
        'Rango@k',
        'RangoRec@k',
    ]
    metrics = pd.DataFrame(metrics, columns=cols, index=indx)
    st.dataframe(metrics, use_container_width=True)
with col2:
    write('<u>*Validación*</u>')
    metrics = [
        ['-', '-', '-', '-'],
        ['-', '-', '-', '-'],
        ['0.0238', '0', '0.5', '0'],
        ['0.0510', '0', '0.5', '0'],
        ['1.50%', '0%', '20%', '0%'],
        ['13.00%', '0%', '100%', '0%'],
        ['3.9231', '4', '9', '-'],
        ['0.0607', '0', '1', '0'],
    ]
    cols = ['Media', 'Mediana', 'Más alto', 'Más bajo']
    indx = [
        'T. Entrenamiento',
        'T. Evaluación',
        'MAP@k',
        'R@k',
        'P@k',
        'HR@k',
        'Rango@k',
        'RangoRec@k',
    ]
    metrics = pd.DataFrame(metrics, columns=cols, index=indx)
    st.dataframe(metrics, use_container_width=True)


col1, _ = st.columns([8.5, 1])
with col1:
    st.header('📝 Conclusiones 💬')
    write(
        'Al igual que el modelo de **co-popularidad**, una gran parte del tiempo de entrenamiento se invierte en construir el conjunto de sesiones, ya que nuestro dataset no venía preparado '
        'para ello. Aún así, este tiempo vuelve a ser un *one-off*, y ampliar con nuevas sesiones no es tan costoso. De hecho, al ser el modelo una red neuronal de una sola capa, podemos '
        'retomar el entrenamiento en cualquier momento si añadimos más sesiones. Con esto conseguimos que el modelo '
        + highlight('**escale**', colors.HIGHLIGHT_GREEN)
        + ' conforme añadimos más sesiones.'
    )
    write(
        'Teóricamente, Word2Vec también sufre de '
        + highlight('**popularity bias**', colors.HIGHLIGHT_RED)
        + ': los productos populares apareceran con frecuencia "cerca" de cualquier otro producto. Sin embargo, podemos aprovechar un hiperparámetro llamado '
        + highlight('**ventana**', colors.HIGHLIGHT_BLUE)
        + ', que controla cuánto nos podemos alejar de la palabra central hasta dejar de considerarlo "parte del contexto". Si bien esto no los libera por todo del *bias*, conseguimos '
        'no recomendar productos populares pero que no tienen nada que ver con el *prompt*.'
    )
    write(
        'Este modelo es un claro ejemplo de <u>*Collaborative Filtering*</u>: hemos aprendido qué productos son similares a partir de sus interacciones con los usuarios. Sin embargo, '
        'notamos si bien aprende del conjunto de usuarios, no es capaz de ajustarse a ninguno en particular.'
    )
    write(
        'Además, a pesar de ser capaz de poder ampliar el "vocabulario" según sea necesario, no es capaz de dar recomendaciones para un producto que aún no conoce: sufre de '
        + highlight('**Cold Start**', colors.HIGHLIGHT_RED)
        + ' del lado de los productos, pero no de los usuarios. Por tanto, este modelo es útil para recomendaciones "anónimas", pero fundadas.'
    )

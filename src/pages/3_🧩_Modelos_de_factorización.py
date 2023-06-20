import pandas as pd
import streamlit as st

import utils as colors
from utils import bullet_item, highlight, setup_page, vspace, write

# page configs
setup_page()

# page content
st.title('Modelos basados en factorización')
vspace(3)
col1, _ = st.columns([8, 1])
with col1:
    write(
        'La idea del *embedding* que hemos sacado del modelo de NLP resulta ser central en la mayoría de modelos actuales. Hemos conseguido "codificar" productos abstractos en números, '
        'de forma que productos similares corresponden a puntos cercanos. Sin embargo, con el modelo anterior solo conseguimos codificar *productos*. Con los '
        '<u>*modelos de factorización*</u> (MF) llegamos a un embedding **común** de usuarios y productos.'
    )
    write(
        'Supongamos que tenemos una <u>matriz de valoraciones</u> de tamaño **# usuarios** x **# productos**. Dependiendo del tipo de conjunto de datos que tengamos, estas valoraciones pueden '
        'ser '
        + highlight('**explícitas**', colors.HIGHLIGHT_GRAY)
        + ' (rating del 1 al 5 del usuario al producto), o '
        + highlight('**implícitas**', colors.HIGHLIGHT_GRAY)
        + ' (número de interacciones, probabilidad de interacción...). Similarmente a la matriz de co-ocurrencias, esta es una matriz *dispersa*; rellenar sus huecos equivale a deducir '
        'el interés que puede tener cada usuario por cada producto. Para ello, tomamos un número *k* y **factorizamos** la matriz en dos:'
    )
    bullet_item(
        'Una matriz de tamaño **# usuarios** x **k**, que sirve de *embedding* para los usuarios.'
    )
    bullet_item(
        'Una matriz de tamaño **k** x **# productos**, que sirve de *embedding* para los productos.'
    )
    write(
        'Así, reconstruir la matriz de valoraciones equivale a encontrar el producto escalar (también conocido como *cosine similarity* sin normalizar) entre el embedding de cada usuario '
        'y cada producto. Intuitivamente, cada una de las **k** coordenadas representa una '
        + highlight('**característica latente**', colors.HIGHLIGHT_GREEN)
        + ': un "concepto" abstracto al que el modelo asigna un *grado de participación*. Así, para saber cómo de recomendado es un producto a un usuario, basta con saber la afinidad de cada '
        'usuario/producto a cada uno de los "conceptos", y hacer una media ponderada de ambas afinidades. Visualmente:'
    )
vspace(1)
col1, _, col2, _ = st.columns([3.5, 1, 3, 0.8])
with col1:
    vspace(3)
    write('<u>*Factorización de la matriz*</u>')
    st.image('res/images/mf.png', width=700)
with col2:
    st.image('res/images/latent_feature.png', width=500)

vspace(3)
col1, _, col2, _ = st.columns([7, 0.5, 4.5, 1])
with col1:
    st.header('🧩 Recomendador ALS 🔄')
    write(
        'Para una primera implementación, utilizamos **PySpark** junto al método de '
        + highlight('**Alternating Least Squares**', colors.HIGHLIGHT_BLUE)
        + ', que consiste en alternar entre cuál de las dos matrices fijamos, mientras optimizamos la otra. Es uno de los métodos más eficientes para conseguir factorizaciones a gran '
        'escala, y está integrado con el ecosistema de Spark.'
    )
    write(
        'Entre todos los parámetros que ofrece ALS, el que mayor consecuencias tiene es **k**, la *dimensión del espacio de características latentes*. Por ejemplo, si fijamos **k = 1**, nuestro modelo '
        'se reduce a ponderar *popularidades*: cada producto y usuario queda reducido a un único número, que podemos interpretar como "popularidad" para productos, y "afinidad por lo popular" '
        'para usuarios.'
    )
    write(
        'Para evaluar el modelo, bastaría con proporcionarle un usuario y buscar en su correspondiente fila al producto con mayor valoración. De hecho, esto se puede hacer en el sentido '
        'contrario, algo que pocos modelos permiten: a partir de un producto, podemos encontrar los usuarios más afines a él.'
    )
with col2:
    vspace(5)
    st.image('res/images/als.png')
vspace(2)

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
        ['10min', '-', '-', '-'],
        ['10.014s', '10.001s', '10.868s', '9.675s'],
        ['0.1876', '0.1464', '0.6732', '0.0000'],
        ['0.1490', '0.1348', '0.5000', '0.0000'],
        ['26.40%', '20%', '70%', '0%'],
        ['79.00%', '100%', '100%', '0%'],
        ['1.8608', ' 1', '9', '-'],
        ['0.6124', '1', '1', '0'],
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
        ['0.0575', '0', '0.5', '0'],
        ['0.1808', '0', '1', '0'],
        ['6.20%', '0%', '40%', '0%'],
        ['42%', '0%', '100%', '0%'],
        ['5.1905', '5.5', '10', '-'],
        ['0.1316', '0', '1', '0'],
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
        'A pesar de tener una entrada similar al modelo de *co-popularidad*, el modelo '
        + highlight('**escala**', colors.HIGHLIGHT_RED)
        + ' fatal: cada vez que entra un usuario, producto o interacción nueva necesitamos recalcular la factorización por completo.'
    )
    write(
        'Por este mismo motivo, no es capaz de dar recomendaciones para un producto/usuario que aún no conoce: sufre fuertemente de '
        + highlight('**Cold Start**', colors.HIGHLIGHT_RED)
        + '. A cambio, las recomendaciones que da son ajustadas a las preferencias y al historial de cada usuario, en vez de basarse en un "historial común anónimo".'
    )
    write(
        'Contextos como la biología y la química, que no tienen nada que ver con el mundo de las recomendaciones, utilizan este tipo de modelos precisamente por la utilidad que ofrecen las '
        + highlight('**características latentes**', colors.HIGHLIGHT_GREEN)
        + ': son una especie de *clustering* o *clasificación* automática que aprende el modelo, sin ningún tipo de contexto de qué son los productos o los usuarios. Por tanto, una vez '
        'aprendidos, un poco de análisis y exploración nos ayuda a descubrir características sobre productos y usuarios que no son obvias a primera vista.'
    )
    write(
        'Además, este modelo ofrece una forma de recomendar '
        + highlight('**simétrica**', colors.HIGHLIGHT_BLUE)
        + ', algo que pocos otros modelos tienen. Como sumergimos tanto usuarios como productos en un espacio común, podemos:'
    )
    bullet_item(
        ' Encontrar similitud entre dos 👥 <u>*usuarios*</u>, aplicar algoritmos de *clustering* para segmentarlos...'
    )
    bullet_item(
        ' Encontrar similitud entre dos 🛍️ <u>*productos*</u>, aprender nuevas *etiquetas* para clasificar productos...'
    )
    bullet_item(
        ' Recomendar <u>*productos*</u> a un <u>*usuario*</u> ➡️, ordenando su fila en la matriz de valoraciones.'
    )
    bullet_item(
        ' Recomendar <u>*usuarios*</u> a un <u>*producto*</u> ⬅️, ordenando su columna en la matriz de valoraciones.'
    )

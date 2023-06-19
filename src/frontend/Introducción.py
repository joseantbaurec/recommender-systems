"""

Frontend landing page for the interactive demo of RecSys

"""
import streamlit as st
import utils
from utils import bullet_item, highlight, plotly_from_json, setup_page, vspace, write

# page configs
setup_page()

# page content
st.title('Sistemas de Recomendación')
st.write('*... o cómo conocer lo que quieres mejor que tú mismo.*')
vspace(5)

col1, col2, _ = st.columns([1, 2.2, 0.15])
with col1:
    st.image('res/images/mr_recommended.png', width=425)
with col2:
    st.header('📝 El problema 💎')
    vspace(1)
    write(
        'El negocio de casi todas las empresas se puede resumir en ofrecer servicios a sus clientes. Supongamos que nuestro foco no está en la generación de estos servicios, sino '
        'en saber conectar servicios existentes con clientes adecuados: este es el problema que los '
        + highlight('**sistemas de recomendación**', utils.HIGHLIGHT_GRAY)
        + ' intentan resolver.'
    )
    write('Por ejemplo:')
    bullet_item(
        '📢 En <u>*publicidad*</u>, queremos ofrecer nuevas campañas a empresas con los que campañas similares anteriores hayan tenido éxito.'
    )
    bullet_item(
        '🛍️ En <u>*e-commerce*</u>, queremos ofrecer nuevos productos a clientes en base a su historial de compra.'
    )
    bullet_item(
        '🎧 En <u>*música*</u>, queremos ayudar al usuario a descubrir nuevas canciones en función a sus gustos.'
    )
    bullet_item(
        '🍽️ En <u>*restauración*</u>, queremos ayudar a encontrar nuevos restaurantes similares a los que frecuenta el usuario.'
    )
    bullet_item(
        '🎥️ En <u>*multimedia*</u>, queremos recomendar contenido audiovisual conociendo el historial de visualizaciones.'
    )
    vspace(1)
    write(
        'Tal y como los ejemplos nos hacen pensar, la potencia y flexibilidad de los *sistemas de recomendación* recaen en cómo definen el concepto de '
        + highlight('**similitud**', utils.HIGHLIGHT_GRAY)
        + '. Se clasifican en grandes rasgos según el enfoque usado:'
    )
    bullet_item(
        highlight('**Content-based Filtering**', utils.HIGHLIGHT_GREEN)
        + ', donde medimos similitud entre <u>*productos*</u>: si te gustó el producto P y el producto Q '
        'es "similar" a P, entonces te gustará Q.'
    )
    bullet_item(
        highlight('**Collaborative Filtering**', utils.HIGHLIGHT_GREEN)
        + ', donde medimos similitud a través de los <u>*usuarios*</u>: si el usuario X es "similar" al usuario Y, y el producto '
        'P le gustó a Y, entonces le gustará a X.'
    )
    bullet_item(
        highlight('**Hybrid Filtering**', utils.HIGHLIGHT_GREEN)
        + ', un punto intermedio entre los dos métodos anteriores.'
    )
    vspace(1)
    write(
        'La naturaleza del problema nos sugiere dividirlo en dos fases especializadas: una de <u>generación de candidatos</u> o '
        + highlight('**sampling**', utils.HIGHLIGHT_BLUE)
        + ', y otra '
        'de <u>valoración de candidatos</u> o '
        + highlight('**ranking**', utils.HIGHLIGHT_RED)
        + '. Gracias a esta separación podemos reutilizar modelos para otras tareas: modelos de ranking son '
        'útiles para ordenar resultados de búsqueda, y los feeds continuos se benefician de modelos de sampling generosos.'
    )
    vspace(1)

vspace(5)
_, col1, _, col2, _ = st.columns([0.1, 3, 0.15, 1.5, 0.01])
with col1:
    st.header('🥊 Los retos ⚖️')
    write(
        'La situación particular de los sistemas de recomendación no suele ser común en los proyectos de Data Science: la responsabilidad y el éxito del modelo se dividen a partes iguales '
        'entre el '
        + highlight('**usuario**', utils.HIGHLIGHT_BLUE)
        + ', el '
        + highlight('**negocio**', utils.HIGHLIGHT_RED)
        + ' y el '
        + highlight('**modelo**', utils.HIGHLIGHT_GREEN)
        + '. Hay que '
        'tener siempre en cuenta la <u>*experiencia de usuario*</u> que se busca, el poder implantar los <u>*objetivos y valores*</u> del negocio en el modelo, y alcanzar un balance '
        'con la <u>*infraestructura*</u> necesaria para poder operar con el modelo.'
    )
    write(
        'Así, los **retos** a los que nos enfrentamos se convierten en **decisiones** sobre qué medidas tomamos para solventarnos. <u>*Un verdadero sistema de recomendación está compuesto '
        'por varios modelos*</u>, donde cada uno soporta los puntos flacos de los demás: cuantas más herramientas tengamos a nuestra disposición, más robusto será el sistema que construyamos.'
    )
with col2:
    vspace(4)
    st.image('res/images/players.png', width=225)
_, col1, _ = st.columns([0.07, 3, 0.5])
with col1:
    write(
        'Uno de los primeros problemas que nos encontramos es el de la '
        + highlight('**densidad**', utils.HIGHLIGHT_GRAY)
        + ' de los datos: la <u>matriz de interacciones</u> entre usuario-producto '
        'es *muy poco densa*. Además, la distribución de usuarios y productos tiene *las colas muy largas*: los productos **populares** representan un pequeño porcentaje del banco de productos, '
        'pero un altísimo porcentaje del total de ventas. Con usuarios, la situación es similar aunque menor. Si dibujamos las distribuciones, esto se vuelve evidente:'
    )
    vspace(2)
col1, _, col2, _ = st.columns([4, 1, 4, 1])
with col1:
    fig = plotly_from_json('res/figures/items_per_user.json')
    st.plotly_chart(fig)
with col2:
    fig = plotly_from_json('res/figures/sales_count.json')
    st.plotly_chart(fig)

vspace(2)
_, col1, _ = st.columns([0.1, 3, 0.5])
with col1:
    write(
        'Además de la **densidad**, muchos otros aspectos de los sistemas de recomendación son objeto de discusión frecuente:'
    )
vspace(1)
_, col1, _, col2, _ = st.columns([0.3, 4, 1, 4, 0.7])
with col1:
    bullet_item(
        '🎯 '
        + highlight('**Precisión**', utils.HIGHLIGHT_GREEN)
        + ', el balance entre acercarnos a lo que el usuario busca o descubrir posibilidades nuevas.'
    )
    bullet_item(
        '💎 '
        + highlight('**Popularity Bias**', utils.HIGHLIGHT_RED)
        + ', recomendar productos populares deja de ser una buena estrategia rápidamente.'
    )
    bullet_item(
        '📈 '
        + highlight('**Escala**', utils.HIGHLIGHT_BLUE)
        + ', cómo de frecuente es la actualización del modelo, la entrada de nuevos productos...'
    )
with col2:
    bullet_item(
        '❄️ '
        + highlight('**Cold Start**', utils.HIGHLIGHT_RED)
        + ', es difícil hacer recomendaciones personalizadas a un usuario del que no conocemos nada.'
    )
    bullet_item(
        '📥 '
        + highlight('**Fuente del dato**', utils.HIGHLIGHT_BLUE)
        + ', escasa y subjetiva si es explícita, abudante pero difícil de analizar si es implícita.'
    )
    bullet_item(
        '⛏️ '
        + highlight('**Explotación del dato**', utils.HIGHLIGHT_GREEN)
        + ', cuanto más datos tengamos más es posible hacer con ellos: modelos NLP, reconocimiento de imágenes...'
    )
vspace(1)
_, col1, _ = st.columns([0.1, 3, 0.5])
with col1:
    write(
        'En el menú de la derecha podemos explorar los diferentes *modelos base* que pudiéramos utilizar para construir un sistema de recomendación. Evaluamos su *performance*, su *precisión* '
        'y cómo reaccionan ante el re-entrenamiento y el *Cold Start*.'
    )

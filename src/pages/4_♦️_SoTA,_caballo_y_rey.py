import streamlit as st

import utils as colors
from utils import bullet_item, highlight, setup_page, vspace, write

# page configs
setup_page()

# page content
st.title('El estado del arte')
vspace(2)
col1, _, col2, _ = st.columns([7, 0.5, 4, 1])
with col1:
    write(
        'La mayoría de modelos actuales a gran escala utilizan un sistema '
        + highlight('**híbrido**', colors.HIGHLIGHT_GREEN)
        + ' para las recomendaciones: incorporan tanto datos sobre las interacciones, como metadatos de los usuarios y productos. Al trabajar con bancos de productos gigantes y en '
        'constante crecimiento, utilizan <u>*redes neuronales profundas*</u> (DNN) para especializarse en la tarea de '
        + highlight('**generación de candidatos**', colors.HIGHLIGHT_BLUE)
        + ': tratan el problema como un multi-clasificador extremo, y utilizan técnicas de *muestreo negativo* para entrenarlas de manera eficiente.'
    )
    write(
        'De hecho, las empresas más famosas por sus recomendadores (YouTube, Spotify, TikTok, Amazon...), donde el núcleo del negocio es la capacidad de ayudarnos a encontrar contenido en '
        'tiempo real, el éxito llega al alcanzar una infraestructura capaz de soportar el **constante flujo de datos**, más que el propio recomendador en sí. En esos modelos de negocio, '
        'la cantidad de veces que recibimos recomendaciones es tan alta que el modelo se puede permitir *"dar palos de ciego"*.'
    )
with col2:
    vspace(1)
    st.image('res/images/youtube.png')
vspace(2)
_, col1, _, col2, _ = st.columns([0.2, 2, 0.1, 7, 1])
with col1:
    vspace(1)
    st.image('res/images/players.png', width=250)
with col2:
    write(
        'Al final, tal y como comentábamos al principio, el verdadero **State of The Art** depende del caso de negocio que estemos intentando resolver: las grandes compañías pueden '
        'invertir mucho dinero y esfuerzo en tener arquitecturas y modelos novedosos, pero también observan la evolución de su plataforma y sus usuarios.'
    )
    write(
        'Por ejemplo, **YouTube** descubrió en 2016 que no bastaba con "contear" las visitas de los usuarios a los vídeos, sino que el *watchtime* era la verdadera fuente de '
        'información. Esto supuso un cambio radical en la plataforma, y es la que nos ha llevado a tener contenido *clickbait*, a que deje de centrarse en subscripciones a canales, '
        'segmentar mejor su contenido...'
    )
    write(
        'De la mano con la evolución de la tecnología va la '
        + highlight('**regulación**', colors.HIGHLIGHT_GRAY)
        + ': la Unión Europea y EE.UU. han tenido recientemente enfrentamientos con **TikTok**, por la gran influencia que tiene su algoritmo de recomendaciones sobre los adolescentes. '
        'En la UE, exigieron un nivel de '
        + highlight('**transparencia**', colors.HIGHLIGHT_BLUE)
        + ' sin precedentes: TikTok debía informar al usuario de los "motivos" por los que un vídeo había acabado en su *feed*. Más allá de lo legal, esto obliga al modelo a ser capaz '
        'de dar esta información, en lugar de ser una "caja negra".'
    )

vspace(3)
st.header('🔗 Enlaces de interés 🔝')

bullet_item(
    '[Repo para explorar estos modelos y más](https://gitlab.csw-labs.com/data-science/recommendation-systems), por... **mi** 😎.'
)
bullet_item(
    '[Recommender Systems overview](https://github.com/microsoft/recommenders), por **Microsoft**'
)
bullet_item(
    '[YouTube Recommender System breakdown](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf), por **Google** (más técnico), y '
    'un [artículo divulgativo](https://towardsdatascience.com/breaking-down-youtubes-recommendation-algorithm-94aa3aa066c6), más introductorio.'
)
bullet_item(
    ' Un [artículo con el tech stack de TikTok](https://www.lavivienpost.com/how-tiktok-works-architecture-illustrated/), muy general.'
)

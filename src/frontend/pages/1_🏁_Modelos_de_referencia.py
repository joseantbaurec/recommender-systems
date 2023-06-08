import pandas as pd
import streamlit as st
import utils as colors
from utils import bullet_item, highlight, plotly_from_json, setup_page, vspace, write

# page configs
setup_page()

# page content
st.title('Modelos de referencia')
st.write('*El baseline frente al que el resto de modelos se comparan.*')
vspace(5)

col1, _ = st.columns([8.5, 1.5])
with col1:
    st.header('✏️ Recomendador Ad-Hoc 📈')
    vspace(1)
    write(
        'Basándonos en reglas manualmente definidas podemos construir un sistema de recomendación, de relativamente bajo coste y tan cerca a los objetivos del negocio como sean '
        'las propias reglas que lo forman. Por ejemplo:'
    )
    bullet_item(
        'Si no tenemos ningún input ('
        + highlight('**Cold Start**', colors.HIGHLIGHT_BLUE)
        + ' absoluto, no conocemos ni al usuario ni partimos de un producto), no queda otra '
        'que acudir al marketing: destacar productos populares, top vendidos según categoría, novedades...'
    )
    bullet_item(
        'En la fase de '
        + highlight('**ranking**', colors.HIGHLIGHT_RED)
        + ', reglas definidas por el cliente son una buena forma de ordenar las sugerencias proporcionadas por otro '
        'modelo: son flexibles, fáciles de actualizar y sus efectos son rápidamente visibles. Son frecuentes para ordenar los resultados de una *búsqueda*.'
    )
    write(
        'Este método es mucho más frecuente de lo que pueda parecer a simple vista; muchos contextos pueden tener bastante éxito sin necesitar un modelo complejo. Aún así, no '
        'siempre son los mas acertados.'
    )
    write(
        'Precisamente este tipo de métodos nos ayudan a visualizar la importancia del '
        + highlight('**orden**', colors.HIGHLIGHT_GREEN)
        + ' en los que recomendamos los productos. Por '
        'ejemplo, supongamos que damos 100 recomendaciones cada una con una probabilidad del 50% de ser relevantes para el usuario. Para una misma muestra (misma cantidad de aciertos), '
        'la <u>*precisión media ponderada*</u> (en inglés, '
        + highlight('**Mean Average Precision (MAP)**', colors.HIGHLIGHT_GRAY)
        + ') es superior cuanto antes aparezcan las recomendaciones '
        'acertadas:'
    )
    vspace(1)
    fig = plotly_from_json('res/figures/sample_recall_precision.json')
    fig.update_layout(width=1225)
    st.plotly_chart(fig)

vspace(3)
col1, col2, _ = st.columns([3, 8, 1])
with col1:
    vspace(4)
    table = [
        ['-', 2, 0, 0, 3],
        [2, '-', 4, 1, 0],
        [0, 4, '-', 0, 1],
        [0, 1, 0, '-', 0],
        [3, 0, 1, 0, '-'],
    ]
    names = [f'P{i+1}' for i in range(5)]
    table = pd.DataFrame(table, index=names, columns=names).astype(str)
    st.dataframe(table)
with col2:
    st.header('👥 Recomendador de Co-Popularidad 💎')
    write(
        'Si intentamos crear un modelo lo más básico posible pero que incorpore la información sobre el historial de interacciones, llegamos al '
        + highlight('**modelo de co-ocurrencias**', colors.HIGHLIGHT_GREEN)
        + ': construimos una matriz de co-ocurrencias *M* en la que la entrada *(i,j)* responde al número de veces que los productos *i* y *j* han tenido interacción con un mismo usuario.'
    )
    write(
        'Esto nos proporciona una <u>*matriz dispersa*</u> (simétrica) de tamaño *N x N*, si *N = # de productos*. Para obtener recomendaciones a partir de la matriz, necesitamos como input al menos '
        'un producto: tomamos su correspondiente columna en la matriz, y ordenamos por popularidad. Si queremos tener en cuenta más de un producto, o un historial temporal, podemos combinar '
        'las columnas de cada producto a través de una media ponderada, obteniendo así recomendaciones más balanceadas y con más contexto.'
    )
vspace(2)
col1, _ = st.columns([8.5, 1])
with col1:
    st.header('🧮 Estadísticas 💯')
    vspace(1)
    write(
        'Entrenado sobre un dataset de e-commerce donde los usuarios compran productos de distintas categorías, decidimos evaluar el modelo de co-popularidad. Para ser lo más '
        'exhaustivo posible, contabilizamos varias métricas: tiempo de entrenamiento, tiempo de evaluación, precisión, exhaustividad y otras métricas propias de los sistemas '
        'de recomendación.'
    )
    vspace(1)

_, col1, _ = st.columns([0.1, 3, 0.7])
with col1:
    metrics = [
        ['53.288s', '-', '-', '-'],
        ['0.612s', '0.609s', '0.643s', '0.598s'],
        ['0.1374', '0.0742', '0.8521', '0'],
        ['0.1716', '0.1333', '0.75', '0'],
        ['23.3%', '20%', '90%', '0%'],
        ['2.9405', '2', '10', '-'],
        ['0.4695', '0.5', '1', '0'],
    ]
    cols = ['Media', 'Mediana', 'Más alto', 'Más bajo']
    indx = [
        'T. Entrenamiento',
        'T. Evaluación',
        'MAP@k',
        'R@k',
        'P@k',
        'Rango@k',
        'RangoRec@k',
    ]
    metrics = pd.DataFrame(metrics, columns=cols, index=indx)
    st.dataframe(metrics, use_container_width=True)

col1, _ = st.columns([8.5, 1])
with col1:
    write(
        'El tiempo de entrenamiento es un *one-off*, ya que el trabajo duro se realiza la primera vez al construir la matriz. Diferentes implementaciones de matrices dispersas '
        'ayudan a acelerar el tiempo y disminuir el espacio en memoria, pero pueden empeorar el tiempo de consulta (que se mantiene constante en sucesivas peticiones).'
    )
    write(
        'Este modelo, como muchos otros, sufre fuertemente de '
        + highlight('**popularity bias**', colors.HIGHLIGHT_RED)
        + ' y de '
        + highlight('**cold start**', colors.HIGHLIGHT_RED)
        + ': '
        'los productos populares por definición ocurren a la vez que la mayoría, así que es frecuente que aparezcan como sugerencias independientemente del producto. Similarmente, '
        'los productos <u>*nicho*</u> son poco habituales, así que son fácilmente destronados por productos más populares, y si son el prompt para el modelo, suele haber pocos '
        'candidatos disponibles.'
    )
    write(
        'A pesar de todo, el modelo es una referencia base por buenos motivos: la <u>*tasa de acierto*</u> está alrededor del 23%, y el esfuerzo ha sido bastante pequeño. Sin embargo, '
        'no es más que un punto de partida, ya que claramente vemos cómo pasan factura los problemas anteriores: cuando evaluamos el modelo con información nueva, vemos que su '
        'puntuación se queda por los suelos.'
    )
vspace(1)
_, col1, _ = st.columns([0.1, 3, 0.7])
with col1:
    metrics = [
        ['0.0785', '0', '0.6429', '0'],
        ['0.1728', '0', '1', '0'],
        ['4.10%', '0%', '30%', '0%'],
        ['3.3030', '3', '8', '1'],
        ['0.1540', '0', '1', '0'],
    ]
    cols = ['Media', 'Mediana', 'Más alto', 'Más bajo']
    indx = ['MAP@k', 'R@k', 'P@k', 'Rango@k', 'RangoRec@k']
    metrics = pd.DataFrame(metrics, columns=cols, index=indx)
    st.dataframe(metrics, use_container_width=True)

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
        'Entrenado sobre un dataset de e-commerce donde los usuarios compran productos de distintas categorías, decidimos evaluar el modelo de co-popularidad. Para ser lo más exhaustivo '
        'posible, contabilizamos varias métricas: tiempo de entrenamiento, tiempo de evaluación, precisión, exhaustividad y otras métricas propias de los sistemas de recomendación. Una '
        'de las más útiles es el '
        + highlight('**Hit Rate**', colors.HIGHLIGHT_GRAY)
        + ' o *tasa de acierto*: 1 si alguna recomendación es relevante, 0 si no.'
    )
    vspace(1)

col1, _, col2, _ = st.columns([3, 0.1, 3, 0.7])
with col1:
    write('<u>*Entrenamiento*</u>')
    metrics = [
        ['1h 30min', '-', '-', '-'],
        ['0.634s', '0.630s', '0.667s', '0.612s'],
        ['0.1133', '0.0558', '0.7675', '0'],
        ['0.1076', '0.0909', '0.6667', '0'],
        ['18.20%', '15%', '80%', '0%'],
        ['67%', '100%', '100%', '0%'],
        ['2.6567', '2', '10', '-'],
        ['0.4279', '0.2917', '1', '0'],
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
        ['0.0715', '0', '1', '0'],
        ['0.1385', '0', '1', '0'],
        ['4.60%', '0%', '30%', '0%'],
        ['32%', '0%', '100%', '0'],
        ['3.5312', '3', '9', '-'],
        ['0.1539', '0', '1', '0'],
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
        'Vemos como el tiempo de entrenamiento es bastante alto para un modelo tan simple: esto se debe a la construcción de la matriz de co-popularidad, que si bien es muy eficiente '
        'para almacenar (muy poco densa), es costosa de calcular. Sin embargo, una vez calculada, fácilmente podemos añadir nuevas observaciones, así que el modelo '
        + highlight('**escala**', colors.HIGHLIGHT_GREEN)
        + ' bastante bien. Similarmente, las reglas ad-hoc no resultan difíciles de expandir y complementar con nuevas reglas, y pueden aprovechar al máximo todo el '
        + highlight('**conocimiento experto**', colors.HIGHLIGHT_BLUE)
        + ' que se tenga sobre el banco de productos/usuarios.'
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
        'Frecuentemente, utilizamos los modelos **Ad-Hoc** como herramientas de '
        + highlight('**ranking**', colors.HIGHLIGHT_GREEN)
        + ', al ser capaces de inyectar preferencias de negocio en constante cambio de manera eficaz y controlable. En contraste, la mayoría de modelos que expondremos suelen ser '
        'mejor aprovechados como herramientas '
        + highlight('**híbridas**', colors.HIGHLIGHT_GREEN)
        + ' o de '
        + highlight('**sampling**', colors.HIGHLIGHT_GREEN)
        + ', para reducir la búsqueda de candidatos.'
    )

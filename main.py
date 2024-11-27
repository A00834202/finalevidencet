import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patheffects import withStroke
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configuración de la página
st.set_page_config(
    page_title="Comportamiento MC4",
    layout="wide"
)

# Agregar un botón para resaltar el sidebar
st.markdown("""
    <style>
    .css-18e3th9 {
        visibility: hidden;
    }
    .css-1aumxhk {
        visibility: visible;
        position: absolute;
        top: 10px;
        left: 10px;
        font-size: 18px;
        color: #FF6347;
        font-weight: bold;
    }
    </style>
""",unsafe_allow_html=True)


# Simulación de datos
dataset = pd.read_csv('Pacing Dashboard.csv')
dataset = dataset.rename(columns={'Difference Phase 1 (Real - Target)': 'Difference Phase 1'})
def classify_category(value):
    levels = ['soft', 'medium', 'hard']
    return value if value in levels else 'otros'
# Crear una nueva columna con las clasificaciones
dataset['Dureza2'] = dataset['Dureza'].apply(classify_category)
dataset['Date'] = pd.to_datetime(dataset['Date'], format='%d/%m/%y') # Convertir la columna de fechas al formato datetime
dataset['mes'] = dataset['Date'].dt.month # Extraer el mes para usar en el filtro

# Producción
prod = pd.read_excel('Prod_may-ago.xlsx')
prod['Fecha'] = pd.to_datetime(prod['Fecha'], format='%d/%m/%y') # Convertir la columna de fechas al formato datetime
prod['mes'] = prod['Fecha'].dt.month # Extraer el mes para usar en el filtro

# Demoras
demoras = pd.read_excel('SGL_Demoras_Caliente 4 PES.xlsx')
demoras['Fecha'] = pd.to_datetime(demoras['Fecha'], format='%d/%m/%y') # Convertir la columna de fechas al formato datetime
demoras['mes'] = demoras['Fecha'].dt.month # Extraer el mes para usar en el filtro

meses = {5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto'}

# Cargar tipografías
title_font = font_manager.FontProperties(fname='Tradegothicbold.ttf')
body_font = font_manager.FontProperties(fname='Tradegothic.ttf')

# Título y encabezado con imagen
header = st.container()
with header:
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("COMPORTAMIENTO MC4")
    with col2:
        st.image("Ternium_logo.svg.png", width=320)  # Cambia la ruta a tu logotipo

# Sidebar para filtros
st.sidebar.header("Filtros")

mes_seleccionado = st.sidebar.selectbox(
    "Selecciona un mes", list(meses.values())
    )
numero_mes = [k for k, v in meses.items() if v == mes_seleccionado][0]

selected_dureza = st.sidebar.multiselect(
    "Selecciona la Dureza",
    options=dataset["Dureza2"].unique(),
    default=dataset["Dureza2"].unique()
)

# Filtrar los datos según los filtros seleccionados
filtered_dataset = dataset[
    (dataset['mes'] == numero_mes) &
    (dataset["Dureza2"].isin(selected_dureza))
]

prod_filtro = prod[ 
    prod['mes'] == numero_mes
]

demoras_filtro = demoras[
    demoras['mes'] == numero_mes
]

# KPIs
prod_esperada = prod_filtro['Prod_prog'].sum()
prod_realizada = prod_filtro['Prod_real'].sum()
plan_demorados = filtered_dataset.shape[0]
total_demoras = demoras_filtro.shape[0]

# Inserta estilos CSS personalizados para las métricas
st.markdown(
    """
    <style>
    .metric-container {
        display: flex; /* Alinea elementos en fila */
        justify-content: space-around; /* Espacio uniforme entre métricas */
        align-items: center; /* Centra verticalmente los elementos */
        background-color: #f9f9f9; /* Fondo claro */
        padding: 20px; /* Espaciado interno */
        border-radius: 10px; /* Bordes redondeados */
        margin-bottom: 20px; /* Separación inferior */
    }
    .metric {
        text-align: center; /* Centra texto y números */
        font-family: 'Arial', sans-serif; /* Fuente estándar */
    }
    .metric .value {
        font-size: 36px; /* Tamaño grande para los valores */
        font-weight: bold; /* Números en negrita */
        color: #333333; /* Color oscuro para números */
        margin: 0; /* Sin márgenes */
    }
    .metric .label {
        font-size: 16px; /* Tamaño mediano para etiquetas */
        color: #666666; /* Color gris para etiquetas */
        margin-top: 5px; /* Espaciado entre número y etiqueta */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Contenedor de métricas
st.markdown('<div class="metric-container">', unsafe_allow_html=True)

# Añade métricas dinámicas
st.markdown(f"""
    <div class="metric">
        <p class="value">{prod_esperada / 1_000:.1f}k</p>
        <p class="label">Producción Esperada</p>
    </div>
    <div class="metric">
        <p class="value">{prod_realizada / 1_000:.1f}k</p>
        <p class="label">Producción Realizada</p>
    </div>
    <div class="metric">
        <p class="value">{plan_demorados / 1_000:.1f}k</p>
        <p class="label">Planchones Demorados</p>
    </div>
    <div class="metric">
        <p class="value">{total_demoras}</p>
        <p class="label">Paradas / Interrupciones</p>
    </div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)




# Gráficos organizados en grid
grid = st.container()

with grid:
    col1, col2 = st.columns(2)

    # Gráfico 1: Histograma
    with col1:
        furnace_counts = dataset.groupby('Furnace')['ID'].count()
        total_count = furnace_counts.sum()
        percentages = (furnace_counts / total_count * 100).round(2)
        labels = [f"Horno {furnace}" for furnace in furnace_counts.index]
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, furnace_counts.values, color="#01345e")

        # Agregar los porcentajes dentro de las barras
        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height / 2,
                f"{percentage:.0f}%",
                ha='center',
                va='center',
                color='white',
                size=30,
                fontproperties=title_font,
                fontweight='bold'
            )

        plt.title('Distribucion de planchones por Horno', fontproperties=title_font, fontweight='bold', fontsize=50)  # Cambia "fontsize" para el título aquí
        plt.xlabel('', fontsize=25)  # Opcional: Si decides agregar un eje X más descriptivo
        # Eliminar el título del eje Y
        plt.gca().axes.get_yaxis().set_visible(False)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)  # Línea superior
        ax.spines['right'].set_visible(False)  # Línea derecha
        ax.spines['left'].set_visible(False)  # Línea izquierda
        plt.xticks(fontproperties=title_font, size=25)
        plt.tight_layout()
        st.pyplot(plt)


    # Gráfico 2: KDE
    with col2:
        same_hardness_count = (dataset['Dureza'] == dataset['Dureza PC']).sum()
        different_hardness_count = (dataset['Dureza'] != dataset['Dureza PC']).sum()

        # Datos para el gráfico de barras apiladas
        counts = [same_hardness_count, different_hardness_count]
        labels = ['Misma dureza', 'Diferente dureza']
        colors = ['#4DAF4A', '#ea3d30']  # Verde y Rojo
        plt.figure(figsize=(14, 8))
        plt.barh([''], [same_hardness_count], color=colors[0], label='Misma dureza')
        plt.barh([''], [different_hardness_count], left=[same_hardness_count], color=colors[1], label='Diferente dureza')
        total_count = same_hardness_count + different_hardness_count

        # Porcentaje y texto auxiliar para "Misma dureza"
        plt.text(same_hardness_count / 2, 0.05, f"{(same_hardness_count / total_count * 100):.1f}%",
                 ha='center', va='center', color='white', fontweight='bold', fontsize=40, fontproperties=title_font)
        plt.text(same_hardness_count / 2, -0.2, "Misma dureza",
                 ha='center', va='center', color='white', fontweight='bold', fontsize=20, fontproperties=title_font)

        # Porcentaje y texto auxiliar para "Diferente dureza"
        plt.text(same_hardness_count + different_hardness_count / 2, 0.05, f"{(different_hardness_count / total_count * 100):.1f}%",
                 ha='center', va='center', color='white', fontweight='bold', fontsize=40, fontproperties=title_font)
        plt.text(same_hardness_count + different_hardness_count / 2, -0.2, "Diferente dureza",
                 ha='center', va='center', color='white', fontweight='bold', fontsize=20, fontproperties=title_font)
        plt.title('Distribución de Dureza (Misma vs. Diferente)', fontproperties=title_font, fontweight='bold', size=50)
        plt.xlabel('Cantidad de planchones', fontproperties=title_font, fontsize=40)
        plt.xticks(fontsize=25)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)  # Línea superior
        ax.spines['right'].set_visible(False)  # Línea derecha
        ax.spines['left'].set_visible(False)  # Línea izquierda
        plt.tight_layout()
        st.pyplot(plt)

    col3, col4 = st.columns(2)
    with col3:
        # Definir las columnas que representan cada fase
        phases = ['Difference Phase 1', 'Difference Phase 2', 'Difference Phase 3', 'Difference Phase 4', 'Difference Phase 5']
        fig, ax = plt.subplots(figsize=(12, 8))
        # Configurar colores
        inlier_color = '#63656a'  # Dentro del IQR
        outlier_color = '#fab03e'  # Fuera del IQR
        line_color_positive = '#000000'  # Línea promedio positiva
        line_color_negative = '#000000'  # Línea promedio negativa
        mean_values = []

        # Iterar sobre cada fase para graficar los puntos y calcular los promedios
        for i, phase in enumerate(phases, start=1):
            phase_data = dataset[phase].dropna()  # Eliminar valores nulos

            # Calcular el IQR
            q1 = np.percentile(phase_data, 25)
            q3 = np.percentile(phase_data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Clasificar puntos como inliers o outliers
            inliers = phase_data[(phase_data >= lower_bound) & (phase_data <= upper_bound)]
            outliers = phase_data[(phase_data < lower_bound) | (phase_data > upper_bound)]
            ax.scatter([i + np.random.uniform(-0.2, 0.2) for _ in range(len(inliers))], inliers,
                       color=inlier_color, alpha=0.7, label='Dentro del rango' if i == 1 else "")
            ax.scatter([i + np.random.uniform(-0.2, 0.2) for _ in range(len(outliers))], outliers,
                       color=outlier_color, alpha=0.9, label='Fuera del rango' if i == 1 else "")

            mean_value = phase_data.mean()
            mean_values.append(mean_value)

        for j in range(len(mean_values) - 1):
            # Determinar el color de la línea (según el signo del promedio)
            color = line_color_positive if mean_values[j] >= 0 else line_color_negative
            ax.plot([j + 1, j + 2], [mean_values[j], mean_values[j + 1]],
                    color=color, linewidth=2.5, alpha=0.9)

        # Graficar los puntos de los promedios y añadir tags
        for idx, mean_value in enumerate(mean_values, start=1):
            # Graficar el punto del promedio
            ax.scatter(idx, mean_value, color=line_color_positive if mean_value >= 0 else line_color_negative, s=100, zorder=5)
            # Añadir el tag con el valor del promedio
            ax.text(
                idx,
                mean_value + 50,  # Ajustar posición vertical con "+ 50"
                f"{mean_value:.1f}",
                ha='center',
                va='center',
                fontsize=20,
                color='white',
                fontproperties=title_font,
                path_effects=[withStroke(linewidth=3, foreground='black')]  # Bordes negros
            )

        # Personalizar gráfico
        ax.set_xticks(range(1, len(phases) + 1))
        ax.set_xticklabels(['Fase 1', 'Fase 2', 'Fase 3', 'Fase 4', 'Fase 5'], fontproperties=title_font, fontsize=30)
        ax.set_yticklabels(ax.get_yticks(), fontproperties=body_font, fontsize=30)
        ax.set_title('Delta de Temperatura por Fase', fontproperties=title_font, fontsize=50, fontweight='bold')
        ax.set_ylabel('Diferencia (Real - Target)', fontproperties=title_font, fontsize=40)
        ax.axhline(0, color='red', linewidth=2, linestyle='--', alpha=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        legend = ax.legend(
            loc='upper right',
            fontsize=17,
            markerscale=3
        )
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(plt)

    with col4:
        # Calcular estadísticas para Delay Time
        mean_delay = dataset['Delay Time'].mean()
        median_delay = dataset['Delay Time'].median()
        mode_delay = dataset['Delay Time'].mode()[0]  # Si hay múltiples valores, tomar el primero
        plt.figure(figsize=(12, 8)) # Crear el histograma

        # Configuración del color
        blue_color = '#01345e'
        red_color = '#ea3d30'

        n, bins, patches = plt.hist(dataset['Delay Time'], bins=30, edgecolor='black', alpha=0.8) # Crear el histograma y obtener parches

        # Aplicar color azul a todas las barras excepto las últimas dos
        for i, patch in enumerate(patches):
            if bins[i] >= bins[-3]:  # Cambiar color de las últimas dos barras
                patch.set_facecolor(red_color)
            else:
                patch.set_facecolor(blue_color)

        # Mostrar la frecuencia encima de las últimas dos barras
        for patch in patches[-2:]:
            height = patch.get_height()
            if height > 0:  # Solo mostrar si la barra tiene altura
                plt.text(patch.get_x() + patch.get_width() / 2, height + 0.5, f'{int(height)}',
                         ha='center', va='bottom', fontsize=20, fontproperties=body_font, color='black')

        # Agregar líneas para media, mediana y moda
        plt.axvline(mean_delay, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_delay:.2f}')
        plt.axvline(median_delay, color='green', linestyle='-.', linewidth=2, label=f'Mediana: {median_delay:.2f}')
        plt.axvline(mode_delay, color='orange', linestyle='-', linewidth=2, label=f'Moda: {mode_delay:.2f}')
        plt.title('Distribución de Delay Time', fontproperties=title_font, size=50, fontweight='bold')
        plt.xlabel('Delay Time in seconds', fontproperties=title_font, fontsize=40)
        plt.ylabel('Frecuencia', fontproperties=title_font, fontsize=40)
        plt.xticks(fontsize=20)  # Tamaño de los valores en el eje X
        plt.yticks(fontsize=20)  # Tamaño de los valores en el eje Y
        legend = plt.legend(prop=body_font, fontsize=16)
        plt.setp(legend.get_texts(), fontsize=16)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(plt)

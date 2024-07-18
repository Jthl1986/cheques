import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Evolución descuento de cheques",page_icon="💴")

url = "https://raw.githubusercontent.com/Jthl1986/T1/main/iipcJun24.csv"
df1 = pd.read_csv(url, encoding='ISO-8859-1', sep=',')
#
#OCULTAR FUENTE GITHUB
hide_github_link = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("📊 Evolución descuento de cheques")

    # Subir archivo
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    st.markdown("<small>Desarrollado por JSantacecilia - JSaborido - Equipo Agro</small>", unsafe_allow_html=True)

    if uploaded_file:
            try:
                # Leer el archivo CSV sin encabezados
                df0 = pd.read_csv(uploaded_file, header=None)

                # Seleccionar las columnas de interés
                df0 = df0[[4, 16]]
                df0.columns = ['Fecha1', 'Monto Liquidado']
                df0['fecha'] = df0['Fecha1'].str[5:7] + '/' + df0['Fecha1'].str[:4]
                df= pd.merge(df0, df1, on='fecha', how='inner')
                df['Monto Liquidado'] = df['Monto Liquidado'].replace('[\$,]', '', regex=True)  # Eliminar símbolos de moneda y comas
                df['Monto Liquidado'] = pd.to_numeric(df['Monto Liquidado'], errors='coerce').fillna(0).astype(int)
                df['ajustado'] = df['Monto Liquidado']*df['iipc']
                df['ajustado'] = df['ajustado'].astype(int)

                # Convertir 'fecha' a formato datetime DESPUÉS del merge
                df['fecha'] = pd.to_datetime(df['fecha'], format='%m/%Y')

                # Crear un rango de los últimos 24 meses desde el mes actual
                today = pd.to_datetime('today')
                end_date = today.to_period('M')
                start_date = (end_date - 23).to_timestamp()

                all_months = pd.date_range(start=start_date, end=end_date.to_timestamp(), freq='M').to_period('M')

                # Agrupar por mes y sumar los valores ajustados
                df_grouped = df.groupby(df['fecha'].dt.to_period('M')).sum(numeric_only=True)
                df_grouped = df_grouped.reindex(all_months, fill_value=0)

                # Crear gráfico de barras con seaborn
                plt.figure(figsize=(10, 6))
                sns.barplot(x=df_grouped.index.strftime('%B %Y'), y=df_grouped['ajustado'], palette="viridis")

                # Personalizar el gráfico
                plt.xlabel('Meses', fontsize=12)
                plt.ylabel('Totales Ajustados ($)', fontsize=12)  # Añadir la unidad monetaria si aplica
                plt.title('Montos descontados por mes en valores constantes', fontsize=15)
                plt.xticks(rotation=90, ha='right')
                plt.grid(True, axis='y')  # Mostrar líneas de la cuadrícula en el eje y
                plt.ylim(0, df_grouped['ajustado'].max() * 1.1)  # Ajustar límite superior del eje y

                # Añadir etiquetas de valor en cada barra
                for index, value in enumerate(df_grouped['ajustado']):
                    plt.text(index, value + 10, f'{value:,}', ha='center', va='bottom', fontsize=10)

                # Mostrar el gráfico en Streamlit
                st.pyplot()

                with st.expander("Tabla de control"):
                    # Seleccionar columnas a mostrar
                    df['ajustado'] = df['ajustado'].astype(int)
                    columnas_mostrar = ['Mes', 'Monto Liquidado', 'iipc', 'ajustado']
                    # Mostrar el DataFrame con columnas seleccionadas
                    st.dataframe(df.loc[:, columnas_mostrar], width=1000)

            except Exception as e:
             st.error(f"Error al leer el archivo: {e}")

if __name__ == "__main__":
       main()
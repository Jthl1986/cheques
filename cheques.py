import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np  # Importar numpy para manejar NaN

url = "https://raw.githubusercontent.com/Jthl1986/T1/main/iipcDec24vf.csv"
df1 = pd.read_csv(url, encoding='ISO-8859-1', sep=',')

def graficadol(resultado_final):
    # Función para formatear los números grandes en millones
    def millions(x, pos):
        'The two args are the value and tick position'
        return '%1.1fM' % (x * 1e-6)

    # Eliminar el último mes de los datos (si es necesario)
    resultado_final = resultado_final.iloc[:-1]

    # Crear un rango de fechas completo para los últimos 12 meses
    fecha_actual = datetime.now()
    meses_completos = pd.date_range(end=fecha_actual, periods=12, freq='MS').strftime('%Y-%m').tolist()

    # Convertir 'mes_anio_acred' a formato string para facilitar la comparación
    resultado_final['mes_anio_acred'] = resultado_final['mes_anio_acred'].astype(str)

    # Crear un DataFrame con todos los meses
    df_completo = pd.DataFrame({'mes_anio_acred': meses_completos})

    # Combinar con los datos existentes
    resultado_final = pd.merge(df_completo, resultado_final, on='mes_anio_acred', how='left')

    # Rellenar los valores faltantes con 0
    resultado_final['suma_monto'] = resultado_final['suma_monto'].fillna(0)
    resultado_final['cantidad_cheques'] = resultado_final['cantidad_cheques'].fillna(0)
    resultado_final['promedio_diferencia_dias'] = resultado_final['promedio_diferencia_dias'].fillna(np.nan)
    resultado_final['coeficiente_variacion'] = resultado_final['coeficiente_variacion'].fillna(0)

    # Aplicar estilo
    sns.set_style("whitegrid")

    # Crear la figura y el eje principal
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Graficar el monto como barras (eje principal)
    barplot = sns.barplot(x='mes_anio_acred', y='suma_monto', data=resultado_final, palette='Blues', ax=ax1, edgecolor='black', alpha=0.8)

    # Agregar contornos a las barras
    for bar in barplot.patches:
        bar.set_edgecolor('black')

    # Añadir los valores encima de las barras, en millones
    for p in barplot.patches:
        height = p.get_height()  # Obtener la altura de cada barra
        ax1.text(p.get_x() + p.get_width() / 2, height + 5, f'{height / 1e6:.1f}M', 
                ha='center', va='bottom', color='k', fontsize=10)

    # Añadir la cantidad de cheques y el coeficiente de variación dentro de las barras
    for i, p in enumerate(barplot.patches):
        cantidad = resultado_final['cantidad_cheques'].iloc[i]  # Obtener la cantidad de cheques para ese mes
        cv = resultado_final['coeficiente_variacion'].iloc[i]  # Obtener el coeficiente de variación para ese mes
        # Colocar el texto en el interior de la barra
        ax1.text(p.get_x() + p.get_width() / 2, p.get_height() / 2, f'{cantidad}\nCV: {cv:.1f}%', 
                ha='center', va='center', color='black', fontsize=10, fontweight='bold')

    # Crear un eje secundario para la línea de diferencia de días
    ax2 = ax1.twinx()  # Crea un segundo eje y compartido con el eje x

    # Graficar la línea de diferencia de días en el eje secundario
    ax2.plot(resultado_final['mes_anio_acred'].astype(str), resultado_final['promedio_diferencia_dias'], color='r', marker='o', markersize=10, linewidth=2.5, linestyle='-', drawstyle='default')

    # Etiquetas y leyendas
    ax1.set_ylabel('Monto en millones de $', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax2.set_ylabel('Promedio Diferencia de Días', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Etiquetas en el eje x
    ax1.set_xticklabels(resultado_final['mes_anio_acred'].astype(str), rotation=45)

    # Ajustar los límites del eje y para el monto
    ax1.set_ylim(0, max(resultado_final['suma_monto'].max(), resultado_final['promedio_diferencia_dias'].max(skipna=True)) * 1.1)

    # Eliminar las líneas de referencia del eje y de monto
    ax1.yaxis.set_tick_params(which='both', length=0, width=0)  # Quita las marcas de ticks
    ax1.grid(False)  # Desactiva la cuadrícula del eje principal (monto)

    # Título del gráfico
    plt.title('Monto, cantidad y plazos en la acreditación de cheques descontados', fontsize=16)

    # Corregir la leyenda para mostrar solo una línea y una barra
    handles, labels = ax1.get_legend_handles_labels()
    line = plt.Line2D([0], [0], color='r', lw=2.5, marker='o', markersize=10)
    bar = plt.Rectangle((0,0),1,1,fc="green", edgecolor='black')
    ax1.legend([line, bar], ['Promedio de Días entre que se descontó y se acreditó', 'Monto en millones de $'], loc='upper left')

    # Mejorar la estética general del gráfico
    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    st.pyplot(fig)

def main():
    st.title("📊 Evolución descuento de cheques")

    # Subir archivos
    uploaded_file = st.file_uploader("Sube archivo CSV - Descontados - consultaOps", type=["csv"])
    uploaded_file2 = st.file_uploader("Sube archivo CSV - Acreditados - valoresAcreditados", type=["csv"])
    st.markdown("<small>Desarrollado por JSantacecilia - JSaborido - Equipo Agro</small>", unsafe_allow_html=True)
    
    df9 = None  # Inicializar df9 fuera de los bloques condicionales

    # Procesar primer gráfico (solo descontados)
    if uploaded_file:
        try:
            # Leer el archivo una sola vez
            df9 = pd.read_csv(uploaded_file, header=None, encoding='latin1')

            # Procesar datos para el primer gráfico
            df0 = df9[[4, 16]]
            df0.columns = ['Fecha1', 'Monto Liquidado']
            df0['fecha'] = df0['Fecha1'].str[5:7] + '/' + df0['Fecha1'].str[:4]
            df = pd.merge(df0, df1, on='fecha', how='inner')
            df['Monto Liquidado'] = df['Monto Liquidado'].replace('[\$,]', '', regex=True)
            df['Monto Liquidado'] = pd.to_numeric(df['Monto Liquidado'], errors='coerce').fillna(0).astype(int)
            df['ajustado'] = df['Monto Liquidado'] * df['iipc']
            df['ajustado'] = df['ajustado'].astype(int)
            df['fecha'] = pd.to_datetime(df['fecha'], format='%m/%Y')

            # Crear rango de 24 meses
            today = pd.to_datetime('today')
            end_date = today.to_period('M')
            start_date = (end_date - 23).to_timestamp()
            all_months = pd.date_range(start=start_date, end=end_date.to_timestamp(), freq='M').to_period('M')

            # Agrupar y reindexar
            df_grouped = df.groupby(df['fecha'].dt.to_period('M')).sum(numeric_only=True)
            df_grouped = df_grouped.reindex(all_months, fill_value=0)
            df_grouped['ajustado_mm'] = df_grouped['ajustado'] / 1_000_000

            # Graficar
            plt.figure(figsize=(10, 6))
            sns.barplot(x=df_grouped.index.strftime('%B %Y'), y=df_grouped['ajustado_mm'], palette="viridis")
            plt.xlabel('Meses')
            plt.ylabel('Totales Ajustados (MM$)')
            plt.title('Montos descontados por mes en valores constantes (en millones)')
            plt.xticks(rotation=90, ha='right')
            plt.grid(True, axis='y')
            plt.ylim(0, df_grouped['ajustado_mm'].max() * 1.1)

            for index, value in enumerate(df_grouped['ajustado_mm']):
                plt.text(index, value + 0.1, f'{value:,.0f}', ha='center', va='bottom', fontsize=10)

            st.pyplot()

        except Exception as e:
            st.error(f"Error al procesar el archivo de descontados: {e}")

    # Procesar segundo gráfico (ambos archivos)
    if uploaded_file2 and df9 is not None:  # Asegurar que df9 ya está cargado
        try:
            # Procesar acreditados
            acreditados_df = pd.read_csv(uploaded_file2, header=None, encoding='latin1')
            acreditados_df = acreditados_df.iloc[:-2]
            acreditados_df = acreditados_df.drop(acreditados_df.columns[[0, 1, 2, 3, 5, 7, 8, 9, 14]], axis=1)
            acreditados_df[['condicion', 'monto']] = acreditados_df.iloc[:, 5].str.split(":", expand=True)
            acreditados_df = acreditados_df.drop(acreditados_df.columns[5], axis=1)
            nuevos_nombres = ['tipo', 'id', 'cheque', 'cuit', 'firmante', 'fecha_acreditacion']
            acreditados_df.columns = nuevos_nombres + list(acreditados_df.columns[6:])
            acreditados_df = acreditados_df[acreditados_df['tipo'] != 'CU']
            acreditados_df['fecha_acreditacion'] = pd.to_datetime(acreditados_df['fecha_acreditacion'])
            acreditados_df['id'] = acreditados_df['id'].astype('int64')

            # Procesar descontados (usar df9 ya cargado, NO volver a leer el archivo)
            descontados_df = df9.drop(df9.columns[[1, 3, 5, 6, 7, 8, 9, 11, 13, 19, 20, 21]], axis=1)
            nuevos_nombres1 = ['id', 'tipo1', 'fecha_descuento', 'monto_descuento', 'cheques_ingreso', 
                              'cheques_pendientes', 'monto pendiente', 'monto liquidado', 'resolucion', 'estado']
            descontados_df.columns = nuevos_nombres1
            descontados_df['fecha_descuento'] = pd.to_datetime(descontados_df['fecha_descuento'])
            descontados_df['id'] = descontados_df['id'].astype('int64')

            # Combinar datasets
            resultado_df = acreditados_df.merge(descontados_df, on='id', how='left')
            resultado_df['diferencia_dias'] = (resultado_df['fecha_acreditacion'] - resultado_df['fecha_descuento']).dt.days
            resultado_df['mes_anio_acred'] = resultado_df['fecha_acreditacion'].dt.to_period('M')
            resultado_df['monto'] = resultado_df['monto'].astype(float)

            # Calcular métricas
            resultado_final = resultado_df.groupby('mes_anio_acred').agg(
                suma_monto=('monto', 'sum'),
                promedio_diferencia_dias=('diferencia_dias', 'mean'),
                desviacion_estandar_dias=('diferencia_dias', 'std'),
                cantidad_cheques=('id', 'size')
            ).reset_index()

            resultado_final['coeficiente_variacion'] = (resultado_final['desviacion_estandar_dias'] / resultado_final['promedio_diferencia_dias']) * 100

            # Generar gráfico
            graficadol(resultado_final)

            # Tabla de top 5 firmantes
            top_firmantes = resultado_df.groupby('firmante').agg(
                cantidad_cheques=('id', 'size'),
                monto_total=('monto', 'sum'),
                promedio_diferencia_dias=('diferencia_dias', 'mean')
            ).reset_index().sort_values('cantidad_cheques', ascending=False).head(5)

            # Formatear montos
            top_firmantes['monto_total'] = top_firmantes['monto_total'].apply(lambda x: f'${x:,.0f}')
            top_firmantes['promedio_diferencia_dias'] = top_firmantes['promedio_diferencia_dias'].round(1)

            # Mostrar tabla
            st.subheader("Top 5 Firmantes por Cantidad de Cheques")
            st.dataframe(
                top_firmantes,
                column_order=("firmante", "cantidad_cheques", "monto_total", "promedio_diferencia_dias"),
                use_container_width=True
            )
            with st.expander("Ver detalle operaciones"):
                st.dataframe(resultado_df)

        except Exception as e:
            st.error(f"Error al procesar los archivos: {e}")


if __name__ == "__main__":
    main()

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Cargar datos de IIPC
url = "https://raw.githubusercontent.com/Jthl1986/T1/main/iipcDec24vf.csv"
df1 = pd.read_csv(url, encoding='ISO-8859-1', sep=',')

# Configuración de Streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)

# Función para graficar evolución de montos
def graficadol(resultado_final):
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

    # Rellenar los valores faltantes
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

    # Añadir los valores encima de las barras
    for p in barplot.patches:
        height = p.get_height()
        ax1.text(p.get_x() + p.get_width() / 2, height + 5, f'{height / 1e6:.1f}M', 
                ha='center', va='bottom', color='k', fontsize=10)

    # Añadir la cantidad de cheques y el coeficiente de variación dentro de las barras
    for i, p in enumerate(barplot.patches):
        cantidad = resultado_final['cantidad_cheques'].iloc[i]
        cv = resultado_final['coeficiente_variacion'].iloc[i]
        ax1.text(p.get_x() + p.get_width() / 2, p.get_height() / 2, f'{cantidad}\nCV: {cv:.1f}%', 
                ha='center', va='center', color='black', fontsize=10, fontweight='bold')

    # Eje secundario para la línea de diferencia de días
    ax2 = ax1.twinx()
    ax2.plot(resultado_final['mes_anio_acred'].astype(str), resultado_final['promedio_diferencia_dias'], 
            color='r', marker='o', markersize=10, linewidth=2.5, linestyle='-')

    # Configuración de ejes y etiquetas
    ax1.set_ylabel('Monto en millones de $', color='k')
    ax2.set_ylabel('Promedio Diferencia de Días', color='r')
    ax1.set_xticklabels(resultado_final['mes_anio_acred'].astype(str), rotation=45)
    ax1.set_ylim(0, max(resultado_final['suma_monto'].max(), resultado_final['promedio_diferencia_dias'].max(skipna=True)) * 1.1)
    ax1.yaxis.set_tick_params(which='both', length=0, width=0)
    ax1.grid(False)

    plt.title('Monto, cantidad y plazos en la acreditación de cheques descontados', fontsize=16)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    st.pyplot(fig)

# Función principal
def main():
    st.title("📊 Evolución descuento de cheques")

    # Subir archivos
    uploaded_file = st.file_uploader("Sube archivo CSV - Descontados - consultaOps", type=["csv"])
    uploaded_file2 = st.file_uploader("Sube archivo CSV - Acreditados - valoresAcreditados", type=["csv"])
    st.markdown("<small>Desarrollado por JSantacecilia - JSaborido - Equipo Agro</small>", unsafe_allow_html=True)
    
    df9 = None
    resultado_df = None

    # Procesar descontados
    if uploaded_file:
        try:
            df9 = pd.read_csv(uploaded_file, header=None, encoding='latin1')
            df0 = df9[[4, 16]].copy()
            df0.columns = ['Fecha1', 'Monto Liquidado']
            
            # Limpiar y convertir montos
            df0['Monto Liquidado'] = df0['Monto Liquidado'].replace('[\$,]', '', regex=True).astype(float)
            
            # Ajustar por IIPC
            df0['fecha'] = df0['Fecha1'].str[5:7] + '/' + df0['Fecha1'].str[:4]
            df = pd.merge(df0, df1, on='fecha', how='inner')
            df['ajustado'] = (df['Monto Liquidado'] * df['iipc']).astype(int)
            df['fecha'] = pd.to_datetime(df['fecha'], format='%m/%Y')

            # Crear rango de 24 meses
            today = pd.to_datetime('today')
            end_date = today.to_period('M')
            start_date = (end_date - 23).to_timestamp()
            all_months = pd.date_range(start=start_date, end=end_date.to_timestamp(), freq='M').to_period('M')

            # Agrupar y reindexar
            df_grouped = df.groupby(df['fecha'].dt.to_period('M')).sum(numeric_only=True).reindex(all_months, fill_value=0)
            df_grouped['ajustado_mm'] = df_grouped['ajustado'] / 1_000_000

            # Graficar
            plt.figure(figsize=(10, 6))
            sns.barplot(x=df_grouped.index.strftime('%B %Y'), y=df_grouped['ajustado_mm'], palette="viridis")
            plt.title('Montos descontados por mes en valores constantes (en millones)')
            plt.xticks(rotation=90)
            plt.grid(True, axis='y')
            
            for index, value in enumerate(df_grouped['ajustado_mm']):
                plt.text(index, value + 0.1, f'{value:,.0f}', ha='center', va='bottom')
                
            st.pyplot()

        except Exception as e:
            st.error(f"Error en descontados: {str(e)}")

    # Procesar acreditados y combinar
    if uploaded_file2 and df9 is not None:
        try:
            # Procesar acreditados
            acreditados_df = pd.read_csv(uploaded_file2, header=None, encoding='latin1').iloc[:-2]
            acreditados_df = acreditados_df.drop(acreditados_df.columns[[0, 1, 2, 3, 5, 7, 8, 9, 14]], axis=1)
            acreditados_df[['condicion', 'monto']] = acreditados_df.iloc[:, 5].str.split(":", expand=True)
            acreditados_df = acreditados_df.drop(acreditados_df.columns[5], axis=1)
            nuevos_nombres = ['tipo', 'id', 'cheque', 'cuit', 'firmante', 'fecha_acreditacion']
            acreditados_df.columns = nuevos_nombres + list(acreditados_df.columns[6:])
            acreditados_df = acreditados_df[acreditados_df['tipo'] != 'CU']
            acreditados_df['fecha_acreditacion'] = pd.to_datetime(acreditados_df['fecha_acreditacion'])
            acreditados_df['id'] = acreditados_df['id'].astype('int64')

            # Procesar descontados
            descontados_df = df9.drop(df9.columns[[1, 3, 5, 6, 7, 8, 9, 11, 13, 19, 20, 21]], axis=1)
            nuevos_nombres1 = ['id', 'tipo1', 'fecha_descuento', 'monto_descuento', 'cheques_ingreso', 
                              'cheques_pendientes', 'monto pendiente', 'monto liquidado', 'resolucion', 'estado']
            descontados_df.columns = nuevos_nombres1
            descontados_df['fecha_descuento'] = pd.to_datetime(descontados_df['fecha_descuento'])
            descontados_df['id'] = descontados_df['id'].astype('int64')

            # Combinar y ajustar
            resultado_df = acreditados_df.merge(descontados_df, on='id', how='left')
            resultado_df['diferencia_dias'] = (resultado_df['fecha_acreditacion'] - resultado_df['fecha_descuento']).dt.days
            resultado_df = pd.merge(
                resultado_df,
                df1,
                left_on=resultado_df['fecha_acreditacion'].dt.strftime('%m/%Y'),
                right_on='fecha',
                how='left'
            )
            resultado_df['monto'] = resultado_df['monto'].astype(float)
            resultado_df['monto_ajustado'] = resultado_df['monto'] * resultado_df['iipc']
            resultado_df['mes_anio_acred'] = resultado_df['fecha_acreditacion'].dt.to_period('M')

            # Generar gráfico principal
            resultado_final = resultado_df.groupby('mes_anio_acred').agg(
                suma_monto=('monto_ajustado', 'sum'),
                promedio_diferencia_dias=('diferencia_dias', 'mean'),
                desviacion_estandar_dias=('diferencia_dias', 'std'),
                cantidad_cheques=('id', 'size')
            ).reset_index()
            resultado_final['coeficiente_variacion'] = (resultado_final['desviacion_estandar_dias'] / 
                                                      resultado_final['promedio_diferencia_dias']) * 100
            st.subheader("Cheques acreditados")
            graficadol(resultado_final)

            # Top 5 Firmantes con valores ajustados
            top_firmantes = resultado_df.groupby('firmante').agg(
                cantidad_cheques=('id', 'size'),
                monto_total_ajustado=('monto_ajustado', 'sum'),
                promedio_diferencia_dias=('diferencia_dias', lambda x: round(x.mean()))
            ).reset_index()

            # Mostrar tops
            
            st.subheader("Top 5 por Cantidad")
            top_cantidad = top_firmantes.sort_values('cantidad_cheques', ascending=False).head(5)
            top_cantidad['monto_total_ajustado'] = top_cantidad['monto_total_ajustado'].apply(lambda x: f'${x:,.0f}')
            st.dataframe(top_cantidad[['firmante', 'cantidad_cheques', 'monto_total_ajustado', 'promedio_diferencia_dias']])

            st.subheader("Top 5 por Monto Ajustado")
            top_monto = top_firmantes.sort_values('monto_total_ajustado', ascending=False).head(5)
            top_monto['monto_total_ajustado'] = top_monto['monto_total_ajustado'].apply(lambda x: f'${x:,.0f}')
            st.dataframe(top_monto[['firmante', 'cantidad_cheques', 'monto_total_ajustado','promedio_diferencia_dias']])

            # Gráfico de análisis por firmante
            st.subheader("Análisis Temporal por Firmante")
            selected_firmante = st.selectbox(
                "Seleccionar firmante del Top 5:",
                options=top_monto['firmante'].tolist(),
                key="firmante_selector"
            )
            
            if selected_firmante:
                df_filtrado = resultado_df[
                    (resultado_df['firmante'] == selected_firmante) &
                    (resultado_df['diferencia_dias'].notna())
                ]
                
                if not df_filtrado.empty:
                    df_evolucion = df_filtrado.groupby(
                        df_filtrado['fecha_acreditacion'].dt.to_period('M')
                    )['diferencia_dias'].mean().reset_index()
                    
                    df_evolucion['fecha'] = df_evolucion['fecha_acreditacion'].dt.to_timestamp()
                    
                    plt.figure(figsize=(12, 6))
                    sns.lineplot(
                        data=df_evolucion,
                        x='fecha',
                        y='diferencia_dias',
                        marker='o',
                        linewidth=2,
                        color='darkblue'
                    )
                    
                    plt.title(f'Evolución Mensual de Plazos de Acreditación - {selected_firmante}')
                    plt.xlabel('Mes')
                    plt.ylabel('Días Promedio')
                    plt.xticks(rotation=45)
                    plt.grid(True, alpha=0.3)
                    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b-%Y'))
                    
                    # Añadir etiquetas de valores
                    for x, y in zip(df_evolucion['fecha'], df_evolucion['diferencia_dias']):
                        plt.text(x, y + 0.5, f'{y:.1f}', ha='center', va='bottom')
                    
                    st.pyplot(plt)
                else:
                    st.warning("No hay datos suficientes para generar el gráfico temporal")

            with st.expander("Ver detalle completo de operaciones"):
                st.dataframe(resultado_df)

        except Exception as e:
            st.error(f"Error en procesamiento: {str(e)}")

if __name__ == "__main__":
    main()
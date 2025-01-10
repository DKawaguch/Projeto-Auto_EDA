import streamlit as st
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from fpdf import FPDF
from io import BytesIO


# Importar Dados
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    elif file.name.endswith('.json'):
        return pd.read_json(file)
    else:
        st.error("Formato de arquivo não suportado. Use CSV, Excel ou JSON.")
        return None
    
# Selecionar as variáveis com base em variância, percentual de missings, e correlação de Pearson
def data_filter(df, missing_threshold, missing_strat, variance_threshold, correlation_threshold):

    # Filtrar as variáveis com alta porcentagem de valores faltantes
    df_filtered = df.loc[:, df.isnull().mean() <= missing_threshold]

    df_filled = df_filtered.copy()

    # Separar variáveis numéricas e categóricas
    num_cols = df_filtered.select_dtypes(include='number').columns
    cat_cols = df_filtered.select_dtypes(include='object').columns

    # Substituir valores numéricos pela média e categóricos por "NA"
    inputer_num = SimpleImputer(strategy=str(missing_strat))
    df_filled[num_cols] = inputer_num.fit_transform(df_filtered[num_cols])
    
    inputer_cat = SimpleImputer(strategy='constant', fill_value='Não Informado')
    df_filled[cat_cols] = inputer_cat.fit_transform(df_filtered[cat_cols])

    # Excluir variáveis com variânica abaixo do limite
    selector = VarianceThreshold(threshold=variance_threshold)
    df_filled[num_cols] = selector.fit_transform(df_filled[num_cols])

    low_variance_cats = [col for col in cat_cols if df_filled[col].value_counts(normalize=True).var() < variance_threshold]
    df_filled.drop(columns=low_variance_cats)

    # Excluir variáveis altamente correlacionadas
    corr_matrix = df_filled.corr().abs()
    upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    to_drop = [column for column in corr_matrix.columns if any(corr_matrix[column][upper_triangle] > correlation_threshold)]
    df_filled = df_filled.drop(columns=to_drop, inplace=True)

    return df_filled

# Apresentar análise de forma gráfica contendo gráficos apropriados para variáveis numéricas e categóricas.
def cat_analysis(df):

    cat_cols = df.select_dtypes(include='object').columns

    # Criar grid para layout dos gráficos
    rows = math.ceil(len(cat_cols) / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))

    # Gráficos para variáveis categóricas
    for i, coluna in cat_cols:
        st.subheader(f"Análise da variável: {coluna}")
        ax=axes[i // 3, i % 3]
        sns.countplot(data=df, x=coluna, ax=ax)
        ax.plot(df[coluna])
        ax.set_title(coluna)
    st.pyplot(fig)
    plt.close(fig)
    
def num_analysis(df):

    num_cols = df.select_dtypes(include='number').columns

    # Criar grid para layout dos gráficos
    rows = math.ceil(len(num_cols) / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))

    # Gráficos para variáveis numéricas
    for i, coluna in num_cols:
        st.subheader(f"Análise da variável: {coluna}")
        ax=axes[i // 3, i % 3]
        sns.histplot(data=df, x=coluna, ax=ax)
        ax.plot(df[coluna])
        ax.set_title(coluna)
    st.pyplot(fig)
    plt.close(fig)

# Apresentar a relação entre duas variáveis numéricas
def correlation_analysis(df):

    num_cols = df.select_dtypes(include='number').columns

    # Criar grid para layout dos gráficos
    var_combinations = math.comb(len(num_cols), 2)
    rows = math.ceil(var_combinations / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))

    # Flatten axes array for easier indexing
    axes = axes.flatten()

    # Loop through all pairs of numerical variables
    plot_index = 0
    for coluna1 in enumerate(num_cols):
        for coluna2 in enumerate(num_cols):
            if coluna1 != coluna2 and coluna1 < coluna2:
                st.subheader(f"Relação entre {coluna1} e {coluna2}")
                sns.scatterplot(data=df, x=coluna1, y=coluna2, ax=axes[plot_index])
                plot_index += 1

    st.pyplot(fig)
    plt.close(fig)

# Gerar um PDF contendo toda análise visual de forma elegante, como um relatório técnico e que seja possível fazer download deste arquivo.
def PDF_generator(analysis_images):

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Relatório de Análise de Dados', ln=True, align='C')
    pdf.ln(10)

    for title, img_bytes in analysis_images:
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, title, ln=True)
        pdf.ln(10)
        pdf.image(img_bytes, x=10, y=30, w=190)

    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output

# Aplicação Streamlit
st.title("Ferramenta Automatizada para Análise Exploratória de Dados")

uploaded_file = st.file_uploader("Carregue seu arquivo (CSV, Excel ou JSON):", type=['csv', 'xlsx', 'json'])

if uploaded_file:
    df = load_data(uploaded_file)

    if df is not None:
        st.write("Dados carregados com sucesso!", df.head())

        st.sidebar.title("Configurações")
        missing_threshold = st.sidebar.slider("Limite máximo de valores ausentes (%)", 0.0, 1.0, 0.2, 0.01)
        missing_strat = st.sidebar.selectbox("Estratégia de preenchimento de valores faltantes:", ['mean', 'median', 'most_frequent'])
        variance_threshold = st.sidebar.slider("Limite de variância mínima", 0.0, 1.0, 0.01, 0.01)
        correlation_threshold = st.sidebar.slider("Limite de correlação", 0.0, 1.0, 0.8, 0.01)

        df_filtered = data_filter(df, missing_threshold, missing_strat, variance_threshold, correlation_threshold)
        st.write("Dados após filtragem:", df_filtered.head())

        if st.button('Analisar variáveis categóricas'):
            cat_analysis(df_filtered)

        if st.button('Analisar variáveis numéricas'):
            num_analysis(df_filtered)
        
        if st.button('Analisar correlação entre variáveis numéricas'):
            correlation_analysis(df_filtered)

        if st.button('Gerar Relatório PDF'):
            analysis_images = []

            num_cols = df_filtered.select_dtypes(include='number').columns
            for col in num_cols:
                fig, ax = plt.subplots()
                sns.histplot(df_filtered[col].dropna(), kde=True, ax=ax)
                img_bytes = BytesIO()
                fig.savefig(img_bytes, format='png')
                img_bytes.seek(0)
                analysis_images.append((f"Distribuição: {col}", img_bytes))
                plt.close(fig)

            pdf_file = PDF_generator(analysis_images)

            st.download_button(
                label="Baixar Relatório PDF",
                data=pdf_file,
                file_name="relatorio_analise.pdf",
                mime="application/pdf"
            )
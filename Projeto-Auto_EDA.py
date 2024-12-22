import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from fpdf import FPDF
from io import BytesIO

print("Todos os pacotes foram importados com sucesso!")

# Função para carregar dados
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        st.error("Formato de arquivo não suportado. Use CSV ou Excel.")
        return None
    
# Função para filtrar variáveis com base em critérios estatísticos
def filter_variables(df, variance_threshold, missing_threshold, correlation_threshold):
    # Remover variáveis com alta porcentagem de valores ausentes
    df_filtered = df.loc[:, df.isnull().mean() <= missing_threshold]

    # Preencher valores ausentes para análise de variância
    # Separar colunas numéricas e categóricas
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # Imputação para colunas numéricas
    imputer_num = SimpleImputer(strategy='mean')
    df[num_cols] = imputer_num.fit_transform(df[num_cols])

    # Imputação para colunas categóricas
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])


    # Selecionar variáveis com variância acima do limite
    selector = VarianceThreshold(threshold=variance_threshold)
    selector.fit(df_filled)
    df_filtered = df_filtered.loc[:, selector.get_support()]

    # Remover variáveis altamente correlacionadas
    corr_matrix = df_filtered.corr().abs()
    upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    to_drop = [column for column in corr_matrix.columns if any(corr_matrix[column][upper_triangle] > correlation_threshold)]
    df_filtered = df_filtered.drop(columns=to_drop)

# Funções de plotagem

# Plot de variáveis numéricas
def plot_numeric_variable(df):
    numeric_vars = df.select_dtypes(include=['float64', 'int64']).columns
    for var in numeric_vars:
        st.subheader(f"Distribuição: {var}")
        fig, ax = plt.subplots()
        sns.histplot(df[var].dropna(), kde=True, ax=ax)
        st.pyplot(fig)

# Plot de vairáveis categóricas
def plot_categorical_variable(df):
    categorical_vars = df.select_dtypes(include=['object', 'category']).columns
    for var in categorical_vars:
        st.subheader(f"Distribuição: {var}")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=var, ax=ax)
        st.pyplot(fig)

# Plot de correlação
def plot_relationship(df, var1, var2):
    st.subheader(f"Relação entre {var1} e {var2}")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=var1, y=var2, ax=ax)
    st.pyplot(fig)

# Função para gerar PDF
def generate_pdf(analysis_images):
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
st.title("Ferramenta de Análise de Dados")

uploaded_file = st.file_uploader("Carregue seu arquivo (CSV ou Excel):")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.write("Dados carregados com sucesso!")
        st.write(df.head())

        st.sidebar.title("Configurações")
        variance_threshold = st.sidebar.slider("Limite de variância mínima", 0.0, 1.0, 0.01, 0.01)
        missing_threshold = st.sidebar.slider("Limite máximo de valores ausentes (%)", 0.0, 1.0, 0.2, 0.01)
        correlation_threshold = st.sidebar.slider("Limite de correlação", 0.0, 1.0, 0.8, 0.01)

        df_filtered = filter_variables(df, variance_threshold, missing_threshold, correlation_threshold)
        st.write("Dados após filtragem:")
        st.write(df_filtered.head())

        st.subheader("Análise Visual")
        st.write("Gráficos para variáveis numéricas:")
        plot_numeric_variable(df_filtered)

        st.write("Gráficos para variáveis categóricas:")
        plot_categorical_variable(df_filtered)

        numeric_vars = df_filtered.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_vars) > 1:
            var1 = st.selectbox("Escolha a primeira variável numérica:", numeric_vars)
            var2 = st.selectbox("Escolha a segunda variável numérica:", numeric_vars)
            if var1 != var2:
                plot_relationship(df_filtered, var1, var2)

        # Gerar PDF
        if st.button("Gerar Relatório PDF"):
            analysis_images = []

            # Salvar gráficos em memória
            numeric_vars = df_filtered.select_dtypes(include=['float64', 'int64']).columns
            for var in numeric_vars:
                fig, ax = plt.subplots()
                sns.histplot(df_filtered[var].dropna(), kde=True, ax=ax)
                img_bytes = BytesIO()
                fig.savefig(img_bytes, format='png')
                img_bytes.seek(0)
                analysis_images.append((f"Distribuição: {var}", img_bytes))
                plt.close(fig)

            pdf_file = generate_pdf(analysis_images)

            st.download_button(
                label="Baixar Relatório PDF",
                data=pdf_file,
                file_name="relatorio_analise.pdf",
                mime="application/pdf"
            )
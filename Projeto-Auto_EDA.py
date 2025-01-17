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
from PIL import Image
import tempfile

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
    num_cols = df_filled.select_dtypes(include='number').columns
    cat_cols = df_filled.select_dtypes(include='object').columns

    # Substituir valores numéricos pela estratégia preferida e categóricos por "NA"
    inputer_num = SimpleImputer(strategy=str(missing_strat))
    df_filled[num_cols] = inputer_num.fit_transform(df_filled[num_cols])
    
    inputer_cat = SimpleImputer(strategy='constant', fill_value='Não Informado')
    df_filled[cat_cols] = inputer_cat.fit_transform(df_filled[cat_cols])

    # Excluir variáveis com variânica abaixo do limite
    selector = VarianceThreshold(threshold=variance_threshold)
    selected_features = selector.fit_transform(df_filled[num_cols])
    selected_columns = df_filled[num_cols].columns[selector.get_support()]
    df_filled = pd.DataFrame(selected_features, columns=selected_columns, index=df_filled.index)
    #df_filled[num_cols] = selector.fit_transform(df_filled[num_cols])

    # Ensure at least one categorical column is retained
    if len(cat_cols) > 0:
        low_variance_cats = [col for col in cat_cols if col in df_filled.columns and df_filled[col].value_counts(normalize=True).var() < variance_threshold]
        if len(low_variance_cats) < len(cat_cols):
            df_filled.drop(columns=low_variance_cats, inplace=True)

    # Excluir variáveis altamente correlacionadas
    corr_matrix = df_filled.corr().abs()
    upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    to_drop = [column for column in corr_matrix.columns if any(corr_matrix.loc[upper_triangle, column] > correlation_threshold)]
    df_filled.drop(columns=to_drop, inplace=True)

    return df_filled

# Apresentar análise de forma gráfica contendo gráficos apropriados para variáveis numéricas e categóricas.
def cat_analysis(df):

    cat_cols = df.select_dtypes(include='object').columns

    # Calcule o número de linhas necessárias
    rows = len(df.select_dtypes(include=['object']).columns) // 3 + 1  # Ajuste conforme necessário

    if rows == 0:
        print("Não há colunas categóricas para plotar.")
        return

    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))

    # Gráficos para variáveis categóricas
    for i, coluna in enumerate(cat_cols):
        #st.subheader(f"Análise da variável: {coluna}")
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
    for i, coluna in enumerate(num_cols):
        #st.subheader(f"Análise da variável: {coluna}")
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
    for i, coluna1 in enumerate(num_cols):
        for j, coluna2 in enumerate(num_cols):
            if i < j:
                #st.subheader(f"Relação entre {coluna1} e {coluna2}")
                sns.scatterplot(data=df, x=coluna1, y=coluna2, ax=axes[plot_index])
                plot_index += 1

    st.pyplot(fig)
    plt.close(fig)

# Gerar um PDF contendo toda análise visual de forma elegante, como um relatório técnico e que seja possível fazer download deste arquivo.
def PDF_generator(images):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    for img_bytes in images:
        pdf.add_page()
        
        # Salvar BytesIO como um arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(img_bytes.getvalue())  # Gravar os dados no arquivo temporário
            temp_file_path = temp_file.name

        # Adicionar a imagem ao PDF
        pdf.image(temp_file_path, x=10, y=30, w=190)

    # Salvar o PDF em um arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
        pdf_output_path = temp_pdf_file.name
        pdf.output(pdf_output_path)  # Grava no arquivo temporário
    
    # Abrir o arquivo PDF temporário e convertê-lo para BytesIO
    with open(pdf_output_path, "rb") as f:
        pdf_output = BytesIO(f.read())

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

            # Análise de variáveis categóricas
            cat_cols = df_filtered.select_dtypes(include='object').columns
            for col in cat_cols:
                fig, ax = plt.subplots()
                sns.countplot(x=df_filtered[col].dropna(), ax=ax)
                img_bytes = BytesIO()
                fig.savefig(img_bytes, format='png')
                img_bytes.seek(0)
                analysis_images.append(img_bytes)
                plt.close(fig)  # Fechar fig após salvar

            # Análise de variáveis numéricas
            num_cols = df_filtered.select_dtypes(include='number').columns
            for col in num_cols:
                fig, ax = plt.subplots()
                sns.histplot(x=df_filtered[col].dropna(), kde=True, ax=ax)
                img_bytes = BytesIO()
                fig.savefig(img_bytes, format='png')
                img_bytes.seek(0)
                analysis_images.append(img_bytes)
                plt.close(fig)  # Fechar fig após salvar

            # Análise de correlação entre variáveis numéricas
            for i, col1 in enumerate(num_cols):
                for j, col2 in enumerate(num_cols):
                    if i < j:
                        fig, ax = plt.subplots()
                        sns.scatterplot(x=df_filtered[col1], y=df_filtered[col2], ax=ax)
                        img_bytes = BytesIO()
                        fig.savefig(img_bytes, format='png')
                        img_bytes.seek(0)
                        analysis_images.append(img_bytes)
                        plt.close(fig)  # Fechar fig após salvar

            # Geração do PDF
            pdf_file = PDF_generator(analysis_images)

            st.download_button(
                label="Baixar Relatório PDF",
                data=pdf_file,
                file_name="relatorio_analise.pdf",
                mime="application/pdf"
            )

            pdf_file = PDF_generator(analysis_images)

            st.download_button(
                label="Baixar Relatório PDF",
                data=pdf_file,
                file_name="relatorio_analise.pdf",
                mime="application/pdf"
            )
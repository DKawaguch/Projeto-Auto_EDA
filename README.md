# Ferramenta Automatizada para Análise Exploratória de Dados

## Descrição do Projeto
Este projeto consiste em uma aplicação desenvolvida com Streamlit para realizar Análise Exploratória de Dados (EDA) de maneira automatizada. A ferramenta permite carregar datasets em diferentes formatos, aplicar filtros baseados em variância, correlação e valores ausentes, e gerar análises visuais detalhadas. Além disso, um relatório técnico em PDF contendo as visualizações geradas pode ser baixado diretamente da aplicação.

## Funcionalidades
- **Carregamento de Dados**:
  - Suporte a arquivos nos formatos `.csv`, `.xlsx` e `.json`.

- **Filtragem de Dados**:
  - Filtragem de colunas com alta proporção de valores ausentes.
  - Exclusão de variáveis com baixa variância.
  - Remoção de variáveis altamente correlacionadas.

- **Análise Visual**:
  - Gráficos apropriados para variáveis categóricas e numéricas.
  - Análise de correlação entre variáveis numéricas por meio de gráficos de dispersão.

- **Geração de Relatório**:
  - Criação de um relatório técnico em PDF com todas as análises visuais realizadas.

## Tecnologias Utilizadas
- **Linguagem de Programação**:
  - Python

- **Bibliotecas**:
  - `Streamlit`: Interface interativa e visual.
  - `pandas`: Manipulação de dados.
  - `numpy`: Operações matemáticas e manipulação de arrays.
  - `matplotlib` e `seaborn`: Criação de gráficos.
  - `scikit-learn`: Seleção de variáveis e imputação de valores ausentes.
  - `fpdf`: Geração de relatórios em PDF.
  - `PIL`: Manipulação de imagens.

## Como Usar
1. **Instale as Dependências**:
   Certifique-se de que as bibliotecas necessárias estão instaladas. Use o comando abaixo para instalar todas as dependências:
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn scikit-learn fpdf pillow
   ```

2. **Execute a Aplicação**:
   No terminal, execute o comando:
   ```bash
   streamlit run Porjeto_Auto-EDA.py
   ```

3. **Carregue o Dataset**:
   Faça o upload do arquivo desejado no formato `.csv`, `.xlsx` ou `.json`.

4. **Configure os Parâmetros**:
   Use a barra lateral para ajustar:
   - Limite de valores ausentes.
   - Estratégia de preenchimento de valores ausentes.
   - Limite mínimo de variância.
   - Limite máximo de correlação.

5. **Realize as Análises**:
   Escolha entre as opções de análise disponíveis:
   - Variáveis categóricas.
   - Variáveis numéricas.
   - Correlação entre variáveis numéricas.

6. **Baixe o Relatório**:
   Gere um relatório técnico em PDF e faça o download.

## Contribuindo
Contribuições são bem-vindas! Para reportar bugs ou sugerir melhorias, abra uma issue ou envie um pull request neste repositório.

## Licença
Este projeto está licenciado sob a licença MIT. Consulte o arquivo LICENSE para mais informações.

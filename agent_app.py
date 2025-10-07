# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 13:22:38 2025

@author: villa
"""

# ===================== CÓDIGO EDA OTIMIZADO COM REQUISITOS =====================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import zscore
from fpdf import FPDF
from dotenv import load_dotenv
import os, json, gc

# ===================== CONFIGURAÇÃO INICIAL =====================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
st.set_page_config(page_title="Agente EDA Genérico", layout="wide")
st.title("🤖 Agente de Análise de CSV — EDA Genérico (Versão PDF Completo)")

HISTORY_PATH = "agent_history.json"

# ===================== FUNÇÕES AUXILIARES =====================
def load_history():
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(hist):
    hist = hist[-20:]  # manter apenas últimas 20 entradas
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)
    return hist

if "history" not in st.session_state:
    st.session_state.history = load_history()

@st.cache_data(show_spinner=False)
def load_csv(file):
    df = pd.read_csv(file)
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass
    return df

# ===================== CONCLUSÕES AUTOMÁTICAS =====================
def gerar_conclusoes(df):
    num_cols = df.select_dtypes(include=['float64','int64']).columns.tolist()
    conclusions = []

    # Outliers
    z_scores = np.abs(zscore(df[num_cols]))
    outliers_count = (z_scores > 3).sum(axis=0)
    outlier_cols = [col for col, count in zip(num_cols, outliers_count) if count > 0]
    conclusions.append(f"Colunas com outliers significativos: {outlier_cols}" if outlier_cols else "Não foram detectados outliers relevantes.")

    # Tendência central
    high_mean_cols = df[num_cols].mean().sort_values(ascending=False).head(3).index.tolist()
    conclusions.append(f"Colunas com maiores médias: {high_mean_cols}")

    # Correlação
    corr = df[num_cols].corr().abs()
    high_corr_pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().sort_values(ascending=False)
    if not high_corr_pairs.empty:
        conclusions.append(f"Maiores correlações entre variáveis: {high_corr_pairs.head(3).to_dict()}")
    else:
        conclusions.append("Não há correlações fortes entre as variáveis.")

    # Valores extremos de transações
    if 'Amount' in df.columns:
        high_amounts = df['Amount'].sort_values(ascending=False).head(5).tolist()
        conclusions.append(f"Maiores valores de transação: {high_amounts}")

    return "\n".join(conclusions)

# ===================== GERAÇÃO DE PDF =====================
def gerar_pdf(hist, conclusoes=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Agentes Autônomos – Relatório da Atividade Extra", ln=True, align="C")
    pdf.ln(10)

    def write_text(text, bold=False, size=11):
        pdf.set_font("Arial", "B" if bold else "", size)
        pdf.multi_cell(0, 7, str(text))

    # Framework e Estrutura
    write_text("Framework: Streamlit + Python + pandas + seaborn + matplotlib + sklearn", bold=True)
    write_text("Estrutura da solução: Upload CSV -> Perguntas EDA -> Conclusões -> PDF completo\n", bold=True)

    # 4 Perguntas + respostas (uma com gráfico)
    questions_to_include = hist[:4] if len(hist) >= 4 else hist
    for h in questions_to_include:
        write_text(f"Pergunta: {h['query']}", bold=True)
        write_text(f"Resposta: {h['result'][:700]}")

    # Conclusões
    if conclusoes:
        write_text("\nPergunta: Conclusões do agente", bold=True)
        write_text(conclusoes)

    pdf_bytes = pdf.output(dest='S').encode('latin-1', errors='ignore')
    return pdf_bytes

# ===================== INTERFACE PRINCIPAL =====================
st.sidebar.header("📘 Instruções")
st.sidebar.markdown("""
1. Carregue um CSV  
2. Faça perguntas sobre o dataset  
3. Ajuste número de clusters, se desejar  
4. O agente responde com análise objetiva  
5. Gere conclusões e exporte o PDF  
""")

uploaded_file = st.file_uploader("📂 Carregue o CSV", type="csv")

if uploaded_file:
    df = load_csv(uploaded_file)
    st.success(f"CSV carregado! Formato: {df.shape}")

    MAX_SAMPLE = 50000
    df_sample = df.sample(MAX_SAMPLE, random_state=42) if len(df) > MAX_SAMPLE else df

    numerical_columns = df_sample.select_dtypes(include=['float64','int64']).columns.tolist()
    categorical_columns = df_sample.select_dtypes(include=['object']).columns.tolist()

    query = st.text_input("Faça sua pergunta de EDA:")

    n_clusters = st.sidebar.number_input("Número de clusters (KMeans)", min_value=2, max_value=10, value=3, step=1)

    if query:
        st.info("🤖 Gerando análise objetiva...")
        query_lower = query.lower()
        result = ""

        # Tipos de dados
        if "tipo" in query_lower or "categoria" in query_lower:
            result = f"Colunas numéricas: {numerical_columns}\nColunas categóricas: {categorical_columns}"

        # Distribuição
        elif "distribuição" in query_lower:
            for col in numerical_columns:
                fig, ax = plt.subplots(figsize=(4,3))
                sns.histplot(df_sample[col].dropna(), bins=10, kde=True, ax=ax)
                ax.set_title(col)
                st.pyplot(fig)
                plt.close(fig)
                gc.collect()
            # Cross-tabs automáticas para categóricas
            cross_tabs = ""
            for col in categorical_columns:
                cross_tabs += f"\nColuna {col} - Contagem:\n{df_sample[col].value_counts().to_dict()}"
            result = cross_tabs

        # Tendência temporal automática
        elif "temporal" in query_lower or "tendência" in query_lower:
            date_cols = df_sample.select_dtypes(include=['datetime64','object']).columns.tolist()
            for col in date_cols:
                try:
                    df_sample[col] = pd.to_datetime(df_sample[col])
                    if numerical_columns:
                        fig, ax = plt.subplots(figsize=(10,5))
                        ax.plot(df_sample[col], df_sample[numerical_columns[0]])
                        ax.set_title(f"Tendência temporal: {numerical_columns[0]} x {col}")
                        st.pyplot(fig)
                        plt.close(fig)
                        gc.collect()
                        result = f"Gráfico temporal gerado para {col} x {numerical_columns[0]}"
                        break
                except Exception:
                    continue
            if not result:
                result = "Nenhuma coluna de data/hora detectada para análise temporal."

        # Clusters com K ajustável
        elif "cluster" in query_lower or "agrupamento" in query_lower:
            if numerical_columns:
                try:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(df_sample[numerical_columns])
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_scaled)
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    fig, ax = plt.subplots(figsize=(8,6))
                    scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=kmeans.labels_, cmap='viridis', alpha=0.6)
                    ax.set_title(f"Clusters via PCA + KMeans (K={n_clusters})")
                    st.pyplot(fig)
                    plt.close(fig)
                    gc.collect()
                    result = f"Clusters gerados com K={n_clusters}"
                except Exception as e:
                    result = f"Erro ao gerar clusters: {e}"

        else:
            result = "Pergunta não reconhecida ou não implementada para análise objetiva."

        st.subheader("Resultado da Análise")
        st.text(result)
        st.session_state.history.append({"query": query, "result": result})
        st.session_state.history = save_history(st.session_state.history)

    st.markdown("---")

    if st.button("📄 Gerar Relatório PDF"):
        conclusoes_text = gerar_conclusoes(df_sample)
        pdf_bytes = gerar_pdf(st.session_state.history, conclusoes_text)
        st.download_button(
            "Baixar Relatório PDF",
            data=pdf_bytes,
            file_name="Agentes_Autonomos_Relatorio.pdf",
            mime="application/pdf"
        )

    st.markdown("### Visualizar parte do dataset")
    st.dataframe(df_sample.head(200))

else:
    st.info("💡 Carregue um CSV para começar.")

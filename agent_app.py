# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 13:22:38 2025

@author: villa
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io, base64, os, json, gc
from fpdf import FPDF
from dotenv import load_dotenv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import zscore

# =====================
# CONFIGURAÇÃO INICIAL
# =====================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
st.set_page_config(page_title="Agente EDA Genérico", layout="wide")
st.title("🤖 Agente de Análise de CSV — EDA Genérico (Versão Otimizada)")

HISTORY_PATH = "agent_history.json"

# =====================
# FUNÇÕES AUXILIARES
# =====================
def load_history():
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(hist):
    # manter apenas últimas 20 entradas para economia de memória
    hist = hist[-20:]
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

def gerar_pdf(hist):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Agentes Autônomos – Relatório da Atividade Extra", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)

    for h in hist:
        pdf.set_font("Arial", "B", 12)
        pdf.multi_cell(0, 7, f"Pergunta: {h['query']}")
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 7, f"Resposta: {h['result'][:700]}")
        pdf.ln(4)

    path = "Agentes_Autonomos_Relatorio.pdf"
    pdf.output(path)
    return path

# =====================
# INTERFACE PRINCIPAL
# =====================
st.sidebar.header("📘 Instruções")
st.sidebar.markdown("""
1. Carregue um CSV
2. Faça perguntas sobre o dataset
3. O agente responde com análise objetiva
4. Gere conclusões e exporte o PDF
""")

uploaded_file = st.file_uploader("📂 Carregue o CSV", type="csv")

if uploaded_file:
    df = load_csv(uploaded_file)
    st.success(f"CSV carregado! Formato: {df.shape}")

    # Amostragem para datasets grandes
    MAX_SAMPLE = 50000  # reduzido para memória
    df_sample = df.sample(MAX_SAMPLE, random_state=42) if len(df) > MAX_SAMPLE else df

    numerical_columns = df_sample.select_dtypes(include=['float64','int64']).columns.tolist()
    categorical_columns = df_sample.select_dtypes(include=['object']).columns.tolist()

    query = st.text_input("Faça sua pergunta de EDA:")

    if query:
        st.info("🤖 Gerando análise objetiva...")

        # DESCRIÇÃO DOS DADOS
        if "tipo" in query.lower() or "categoria" in query.lower():
            result = f"Colunas numéricas: {numerical_columns}\nColunas categóricas: {categorical_columns}"

        elif "distribuição" in query.lower():
            result = ""
            for col in numerical_columns:
                counts, bins = np.histogram(df_sample[col].dropna(), bins=10)
                result += f"\nColuna {col} - Contagem por bin: {list(counts)}"
            for col in categorical_columns:
                result += f"\nColuna {col} - Contagem por categoria:\n{df_sample[col].value_counts().to_dict()}"

        elif "intervalo" in query.lower() or "mínimo" in query.lower() or "máximo" in query.lower():
            result = df_sample[numerical_columns].agg(['min','max']).to_string()

        elif "tendência central" in query.lower() or "média" in query.lower() or "mediana" in query.lower():
            result = df_sample[numerical_columns].agg(['mean','median']).to_string()

        elif "variabilidade" in query.lower() or "desvio padrão" in query.lower() or "variância" in query.lower():
            result = df_sample[numerical_columns].agg(['std','var']).to_string()

        # PADRÕES E TENDÊNCIAS
        elif "padrões" in query.lower() or "tendências temporais" in query.lower():
            if 'Time' in df_sample.columns:
                fig, ax = plt.subplots(figsize=(10,5))
                ax.plot(df_sample['Time'], df_sample['Amount'])
                ax.set_title('Tendência temporal de Amount')
                ax.set_xlabel('Time')
                ax.set_ylabel('Amount')
                st.pyplot(fig)
                plt.close(fig)
                gc.collect()
                result = "Gráfico de tendência temporal gerado."
            else:
                result = "Coluna 'Time' não encontrada."

        elif "valores mais frequentes" in query.lower() or "menos frequentes" in query.lower():
            result = ""
            for col in df_sample.columns:
                vc = df_sample[col].value_counts()
                result += f"\nColuna: {col}\nMais frequentes: {vc.head(5).to_dict()}\nMenos frequentes: {vc.tail(5).to_dict()}"

        elif "clusters" in query.lower() or "agrupamentos" in query.lower():
            if len(df_sample) > 10000:
                result = "Dataset grande demais para clusterização; reduza a amostra."
            else:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_sample[numerical_columns])
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
                fig, ax = plt.subplots()
                ax.scatter(X_pca[:,0], X_pca[:,1], c=kmeans.labels_)
                ax.set_title("Clusters PCA")
                st.pyplot(fig)
                plt.close(fig)
                gc.collect()
                result = "Clusters gerados usando PCA e KMeans."

        # DETECÇÃO DE ANOMALIAS
        elif "valores atípicos" in query.lower() or "outliers" in query.lower():
            z_scores = np.abs(zscore(df_sample[numerical_columns]))
            outliers_count = (z_scores > 3).sum(axis=0)
            result = f"Outliers por coluna:\n{dict(zip(numerical_columns, outliers_count))}"

        elif "afetam a análise" in query.lower():
            z_scores = np.abs(zscore(df_sample[numerical_columns]))
            df_no_outliers = df_sample[(z_scores < 3).all(axis=1)]
            result = f"Antes:\n{df_sample[numerical_columns].describe().T}\n\nDepois (sem outliers):\n{df_no_outliers[numerical_columns].describe().T}"

        elif "removidos" in query.lower() or "transformados" in query.lower() or "investigados" in query.lower():
            result = "Recomenda-se: remover outliers extremos, transformar variáveis ou investigar casos específicos."

        # RELAÇÕES ENTRE VARIÁVEIS
        elif "relacionadas" in query.lower() or "dispersão" in query.lower():
            subset_cols = numerical_columns[:5]  # limitar pairplot
            fig = sns.pairplot(df_sample[subset_cols])
            st.pyplot(fig)
            plt.close(fig)
            gc.collect()
            result = "Pairplot gerado (apenas primeiras 5 colunas numéricas)."

        elif "correlação" in query.lower():
            corr = df_sample[numerical_columns].corr()
            fig, ax = plt.subplots(figsize=(12,8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)
            plt.close(fig)
            gc.collect()
            result = "Heatmap de correlação gerado."

        elif "influência" in query.lower():
            corr = df_sample[numerical_columns].corr().abs().sum().sort_values(ascending=False)
            result = f"Variáveis com maior influência:\n{corr.to_string()}"

        else:
            result = "Pergunta não reconhecida ou não implementada para análise objetiva."

        st.subheader("Resultado da Análise")
        st.text(result)
        st.session_state.history.append({"query": query, "result": result})
        st.session_state.history = save_history(st.session_state.history)

    st.markdown("---")

    if st.button("📄 Gerar Relatório PDF"):
        path = gerar_pdf(st.session_state.history)
        with open(path, "rb") as f:
            st.download_button("Baixar Relatório PDF", data=f, file_name=path, mime="application/pdf")

    st.markdown("### Visualizar parte do dataset")
    st.dataframe(df_sample.head(200))

else:
    st.info("💡 Carregue um CSV para começar.")

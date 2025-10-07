# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 13:22:38 2025

@author: villa
"""

# ===================== CÓDIGO EDA OTIMIZADO =====================
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
st.title("🤖 Agente de Análise de CSV — EDA Genérico (Versão Otimizada)")

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

# ===================== NOVA FUNÇÃO: CONCLUSÕES AUTOMÁTICAS =====================
def gerar_conclusoes(df):
    num_cols = df.select_dtypes(include=['float64','int64']).columns.tolist()
    conclusions = []

    # 1. Valores extremos
    z_scores = np.abs(zscore(df[num_cols]))
    outliers_count = (z_scores > 3).sum(axis=0)
    outlier_cols = [col for col, count in zip(num_cols, outliers_count) if count > 0]
    if outlier_cols:
        conclusions.append(f"As colunas {outlier_cols} possuem outliers significativos que podem afetar a análise.")
    else:
        conclusions.append("Não foram detectados outliers relevantes.")

    # 2. Tendência central
    high_mean_cols = df[num_cols].mean().sort_values(ascending=False).head(3).index.tolist()
    conclusions.append(f"As colunas com maiores médias são: {high_mean_cols}")

    # 3. Correlação
    corr = df[num_cols].corr().abs()
    high_corr_pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr_pairs = high_corr_pairs.stack().sort_values(ascending=False)
    if not high_corr_pairs.empty:
        top_corr = high_corr_pairs.head(3)
        conclusions.append(f"Maiores correlações entre variáveis: {top_corr.to_dict()}")
    else:
        conclusions.append("Não há correlações fortes entre as variáveis.")

    # 4. Valores extremos de transações (para dataset de fraude)
    if 'Amount' in df.columns:
        high_amounts = df['Amount'].sort_values(ascending=False).head(5).tolist()
        conclusions.append(f"Maiores valores de transação detectados: {high_amounts}")

    return "\n".join(conclusions)

# ===================== CORREÇÃO: GERAÇÃO DE PDF =====================
def gerar_pdf(hist, conclusoes=None):
    """
    Gera PDF em memória (bytes). Usa fonte Unicode local se disponível.
    Se não houver fonte Unicode disponível localmente, faz fallback para latin-1 safe.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # caminhos locais possíveis para fontes Unicode (não faz downloads)
    possible_font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/local/share/fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "C:\\Windows\\Fonts\\DejaVuSans.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]

    font_path = None
    for p in possible_font_paths:
        if os.path.exists(p):
            font_path = p
            break

    use_unicode_font = False
    if font_path:
        try:
            # registra duas famílias (normal + "bold") usando o mesmo arquivo TTF
            pdf.add_font("DejaVu", "", font_path, uni=True)
            # registra uma família separada para "bold" (mesmo TTF) para evitar usar style 'B'
            pdf.add_font("DejaVu-Bold", "", font_path, uni=True)
            use_unicode_font = True
        except Exception:
            use_unicode_font = False

    # Cabeçalho
    if use_unicode_font:
        pdf.set_font("DejaVu", "", 16)
    else:
        pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Agentes Autônomos – Relatório da Atividade Extra", ln=True, align="C")
    pdf.ln(10)

    # helper para escrever texto com fallback de encoding e família
    def write_text(text, bold=False, size=11):
        if use_unicode_font:
            family = "DejaVu-Bold" if bold else "DejaVu"
            pdf.set_font(family, "", size)
            safe_text = str(text)
        else:
            style = "B" if bold else ""
            pdf.set_font("Arial", style, size)
            # garante que caracteres problemáticos não explodam o FPDF
            safe_text = str(text).encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 7, safe_text)

    # escreve histórico de perguntas e respostas
    for h in hist:
        write_text(f"Pergunta: {h['query']}", bold=True, size=12)
        write_text(f"Resposta: {h['result'][:700]}", bold=False, size=11)
        pdf.ln(4)

    if conclusoes:
        write_text("Pergunta: Conclusões do agente", bold=True, size=12)
        write_text(conclusoes, bold=False, size=11)
        pdf.ln(4)

    # Gera PDF em memória e retorna bytes (encode latin-1 para compatibilidade com FPDF internamente)
    pdf_bytes = pdf.output(dest='S').encode('latin-1', errors='ignore')
    return pdf_bytes

# ===================== INTERFACE PRINCIPAL =====================
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

    MAX_SAMPLE = 50000
    df_sample = df.sample(MAX_SAMPLE, random_state=42) if len(df) > MAX_SAMPLE else df

    numerical_columns = df_sample.select_dtypes(include=['float64','int64']).columns.tolist()
    categorical_columns = df_sample.select_dtypes(include=['object']).columns.tolist()

    query = st.text_input("Faça sua pergunta de EDA:")

    if query:
        st.info("🤖 Gerando análise objetiva...")
        query_lower = query.lower()
        result = ""

        # ==== DESCRIÇÃO, PADRÕES, OUTLIERS, RELAÇÕES ====
        if "tipo" in query_lower or "categoria" in query_lower:
            result = f"Colunas numéricas: {numerical_columns}\nColunas categóricas: {categorical_columns}"

        elif "distribuição" in query_lower:
            n_cols_per_row = 3
            num_cols = len(numerical_columns)
            for i in range(0, num_cols, n_cols_per_row):
                cols = st.columns(n_cols_per_row)
                for j, col in enumerate(numerical_columns[i:i+n_cols_per_row]):
                    with cols[j]:
                        fig, ax = plt.subplots(figsize=(4,3))
                        sns.histplot(df_sample[col].dropna(), bins=10, kde=True, ax=ax)
                        ax.set_title(col, fontsize=10)
                        ax.set_xlabel("")
                        ax.set_ylabel("")
                        st.pyplot(fig)
                        plt.close(fig)
                        gc.collect()
            for col in categorical_columns:
                result += f"\nColuna {col} - Contagem por categoria:\n{df_sample[col].value_counts().to_dict()}"

        elif "intervalo" in query_lower or "mínimo" in query_lower or "máximo" in query_lower:
            result = df_sample[numerical_columns].agg(['min','max']).to_string()

        elif "tendência central" in query_lower or "média" in query_lower or "mediana" in query_lower:
            result = df_sample[numerical_columns].agg(['mean','median']).to_string()

        elif "variabilidade" in query_lower or "desvio padrão" in query_lower or "variância" in query_lower:
            result = df_sample[numerical_columns].agg(['std','var']).to_string()

        elif "padrões" in query_lower or "tendências temporais" in query_lower:
            if 'Time' in df_sample.columns and 'Amount' in df_sample.columns:
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
                result = "Coluna 'Time' ou 'Amount' não encontrada."

        elif "valores mais frequentes" in query_lower or "menos frequentes" in query_lower:
            for col in df_sample.columns:
                vc = df_sample[col].value_counts()
                result += f"\nColuna: {col}\nMais frequentes: {vc.head(5).to_dict()}\nMenos frequentes: {vc.tail(5).to_dict()}"

        elif "clusters" in query_lower or "agrupamentos" in query_lower:
            if len(df_sample) > 10000:
                result = "Dataset grande demais para clusterização; reduza a amostra."
            else:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_sample[numerical_columns])
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
                fig, ax = plt.subplots()
                scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=kmeans.labels_, cmap='viridis', alpha=0.6)
                ax.set_title("Clusters PCA")
                st.pyplot(fig)
                plt.close(fig)
                gc.collect()
                result = "Clusters gerados usando PCA e KMeans."

        elif "valores atípicos" in query_lower or "outliers" in query_lower:
            z_scores = np.abs(zscore(df_sample[numerical_columns]))
            outliers_count = (z_scores > 3).sum(axis=0)
            result = f"Outliers por coluna:\n{dict(zip(numerical_columns, outliers_count))}"

        elif "afetam a análise" in query_lower:
            z_scores = np.abs(zscore(df_sample[numerical_columns]))
            df_no_outliers = df_sample[(z_scores < 3).all(axis=1)]
            result = f"Antes:\n{df_sample[numerical_columns].describe().T}\n\nDepois (sem outliers):\n{df_no_outliers[numerical_columns].describe().T}"

        elif "removidos" in query_lower or "transformados" in query_lower or "investigados" in query_lower:
            result = "Recomenda-se: remover outliers extremos, transformar variáveis ou investigar casos específicos."

        elif "relacionadas" in query_lower or "dispersão" in query_lower:
            subset_cols = numerical_columns[:5]
            pairgrid = sns.pairplot(df_sample[subset_cols])
            st.pyplot(pairgrid.fig)
            plt.close(pairgrid.fig)
            gc.collect()
            result = "Pairplot gerado (apenas primeiras 5 colunas numéricas)."

        elif "correlação" in query_lower:
            corr = df_sample[numerical_columns].corr()
            fig, ax = plt.subplots(figsize=(12,8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)
            plt.close(fig)
            gc.collect()
            high_corr = corr.unstack().sort_values(ascending=False)
            high_corr = high_corr[high_corr < 1]
            top_corr = high_corr[0:5]
            result = f"Heatmap de correlação gerado.\nSim, há correlação significativa entre algumas variáveis, por exemplo:\n{top_corr.to_string()}"

        elif "influência" in query_lower:
            corr_sum = df_sample[numerical_columns].corr().abs().sum().sort_values(ascending=False)
            top_5 = corr_sum.head(5)
            low_5 = corr_sum.tail(5)
            result = f"Variáveis com maior influência:\n{top_5.to_string()}\n\nVariáveis com menor influência:\n{low_5.to_string()}"

        else:
            result = "Pergunta não reconhecida ou não implementada para análise objetiva."

        st.subheader("Resultado da Análise")
        st.text(result)
        st.session_state.history.append({"query": query, "result": result})
        st.session_state.history = save_history(st.session_state.history)

    st.markdown("---")

    # ==== Botão de gerar PDF com conclusões ====
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

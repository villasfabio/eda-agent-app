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
import io, base64, os, json, traceback, contextlib
from openai import OpenAI
from dotenv import load_dotenv
from fpdf import FPDF

# =====================
# CONFIGURAÇÃO INICIAL
# =====================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY não encontrada. Configure em .env ou em Secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
st.set_page_config(page_title="Agente EDA Genérico", layout="wide")
st.title("🤖 Agente de Análise de CSV — EDA Genérico (Versão Final)")

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
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)

if "history" not in st.session_state:
    st.session_state.history = load_history()

@st.cache_data(show_spinner=False)
def load_csv(file):
    df = pd.read_csv(file)
    # Corrige depreciação do errors='ignore'
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass
    return df

def generate_response(prompt, mode="code"):
    """mode='code' retorna código Python; mode='text' retorna resumo textual"""
    system_msg = "Você é um especialista em EDA."
    if mode == "code":
        system_msg += " Gere apenas código Python executável, sem ```."
    else:
        system_msg += " Responda com texto analítico, claro e objetivo."

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt}
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=1200,
        temperature=0
    )
    return resp.choices[0].message.content.strip()

def execute_code(code, df):
    """Executa código Python seguro e captura gráficos e prints"""
    allowed_builtins = {
        "__import__": __import__,
        "print": print,
        "len": len, "min": min, "max": max, "sum": sum,
        "range": range, "int": int, "float": float, "str": str,
        "bool": bool, "list": list, "dict": dict, "set": set, "enumerate": enumerate
    }
    local_env = {"df": df, "pd": pd, "plt": plt, "sns": sns, "px": px, "io": io, "base64": base64, "st": st}
    output = io.StringIO()
    img_b64_list = []

    try:
        with contextlib.redirect_stdout(output):
            exec(code, {"__builtins__": allowed_builtins}, local_env)
    except Exception:
        return f"Erro durante execução:\n{traceback.format_exc()}", None

    text_output = output.getvalue().strip()

    # Captura gráficos Matplotlib
    for fig_num in plt.get_fignums():
        buf = io.BytesIO()
        plt.figure(fig_num).savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_b64_list.append(base64.b64encode(buf.read()).decode())
        plt.close(fig_num)

    if not text_output:
        text_output = "Código executado com sucesso, mas sem saída textual."

    return text_output, img_b64_list if img_b64_list else None

def gerar_pdf(hist):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Agentes Autônomos – Relatório da Atividade Extra", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)

    sections = [
        ("1. Framework escolhida", "Streamlit + OpenAI API (gpt-4o-mini)."),
        ("2. Estrutura da solução", "O agente lê um CSV, interpreta perguntas, gera código Python para EDA com gráficos e conclusões."),
        ("3. Perguntas e respostas", "")
    ]

    for title, content in sections:
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, title, ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 7, content)
        pdf.ln(4)

    for h in hist:
        pdf.set_font("Arial", "B", 12)
        pdf.multi_cell(0, 7, f"Pergunta: {h['query']}")
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 7, f"Resposta: {h['result'][:700]}")
        pdf.ln(4)

    pdf.output("Agentes Autônomos – Relatório da Atividade Extra.pdf")
    return "Agentes Autônomos – Relatório da Atividade Extra.pdf"

# =====================
# INTERFACE PRINCIPAL
# =====================
st.sidebar.header("📘 Instruções")
st.sidebar.markdown("""
1. Carregue um CSV
2. Faça perguntas sobre o dataset
3. O agente responde com código, gráficos e texto
4. Gere conclusões e exporte o PDF
""")

uploaded_file = st.file_uploader("📂 Carregue o CSV", type="csv")

if uploaded_file:
    df = load_csv(uploaded_file)
    st.success(f"CSV carregado! Formato: {df.shape}")

    MAX_SAMPLE = 150000
    df_sample = df.sample(MAX_SAMPLE, random_state=42) if len(df) > MAX_SAMPLE else df

    # Colunas numéricas relevantes
    excluded_cols = ['Time', 'Class']
    numeric_cols = df_sample.select_dtypes(include='number').columns.difference(excluded_cols)
    df_numeric = df_sample[numeric_cols]

    df_info = f"Colunas: {list(df_sample.columns)}; Tipos: {df_sample.dtypes.to_dict()}"
    query = st.text_input("Faça sua pergunta de EDA:")

    if query:
        st.info("🤖 Gerando código e executando automaticamente...")
        prompt = f"O dataframe `df` está carregado com {df_info}. Pergunta: {query}\n" \
                 f"Analise apenas as colunas numéricas relevantes ({list(df_numeric.columns)}). " \
                 f"Inclua gráficos, média, mediana, min, max, std e contagem de valores."
        code = generate_response(prompt, mode="code")
        st.code(code, language="python")
        result, img_b64_list = execute_code(code, df_sample)
        st.subheader("Resultado da Análise")
        st.write(result)
        if img_b64_list:
            for img_b64 in img_b64_list:
                st.image(base64.b64decode(img_b64), use_container_width=True)
        st.session_state.history.append({"query": query, "result": result})
        save_history(st.session_state.history)

    st.markdown("---")

    if st.button("🧠 Gerar Conclusões"):
        with st.spinner("Analisando histórico..."):
            history_txt = "\n".join([f"P: {h['query']}\nR: {h['result']}" for h in st.session_state.history])
            concl = generate_response(f"Com base nestas interações: {history_txt}\nResuma as conclusões gerais sobre o dataset.", mode="text")
        st.subheader("Conclusões do agente")
        st.write(concl)
        st.session_state.history.append({"query": "CONCLUSÕES", "result": concl})
        save_history(st.session_state.history)

    if st.button("📄 Gerar Relatório PDF"):
        path = gerar_pdf(st.session_state.history)
        with open(path, "rb") as f:
            st.download_button("Baixar Relatório PDF", data=f, file_name=path, mime="application/pdf")

    st.markdown("### Visualizar parte do dataset")
    st.dataframe(df_sample.head(200))

    st.markdown("### Ações rápidas pré-definidas")
    if st.button("Resumo estatístico (describe)"):
        st.write(df_numeric.describe().T)
    if st.button("Contagem de classes (Class)"):
        if "Class" in df_sample.columns:
            st.write(df_sample['Class'].value_counts())
            fig = px.histogram(df_sample, x='Class', title='Contagem por Classe')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("A coluna 'Class' não foi encontrada no dataset.")
else:
    st.info("💡 Carregue um CSV para começar.")

import os
import json
import datetime
import psycopg2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# ---------------------------------------------------------
# Leitura manual do .env sem precisar do python-dotenv
# ---------------------------------------------------------
if os.path.exists(".env"):
    with open(".env", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")

# ---------------------------------------------------------
# Configuração da Página
# ---------------------------------------------------------
st.set_page_config(
    page_title="Sistema de Segmentação & Análise de Imagens",
    page_icon="👁️",
    layout="wide"
)

# ---------------------------------------------------------
# Carregamento do Modelo de Segmentação (Cache para performance)
# ---------------------------------------------------------
@st.cache_resource
def load_segmentation_model():
    return YOLO("yolov8n-seg.pt")

model = load_segmentation_model()

# ---------------------------------------------------------
# Funções de Banco de Dados (Neon.tech)
# ---------------------------------------------------------
def get_db_connection(connection_string):
    # Trata a URL para garantir modo Direto e SSL no Neon.tech
    clean_url = connection_string.replace("-pooler.", ".")
    
    if "sslmode=" not in clean_url:
        if "?" in clean_url:
            clean_url += "&sslmode=require"
        else:
            clean_url += "?sslmode=require"
            
    conn = psycopg2.connect(clean_url)
    return conn

def init_db(connection_string):
    conn = get_db_connection(connection_string)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS image_analyses (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255),
            width INT,
            height INT,
            channels INT,
            file_size_kb FLOAT,
            detected_objects JSONB,
            total_objects INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def save_analysis_to_db(connection_string, filename, width, height, channels, file_size_kb, detected_objects, total_objects):
    conn = get_db_connection(connection_string)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO image_analyses 
        (filename, width, height, channels, file_size_kb, detected_objects, total_objects)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (filename, width, height, channels, file_size_kb, json.dumps(detected_objects), total_objects))
    conn.commit()
    cur.close()
    conn.close()

# ---------------------------------------------------------
# Interface Gráfica - Sidebar (Configurações do Banco)
# ---------------------------------------------------------
st.sidebar.title("⚙️ Configurações do Banco")
env_db_url = os.environ.get("DATABASE_URL", "")

db_url = st.sidebar.text_input(
    "String de Conexão Neon.tech (PostgreSQL):",
    value=env_db_url,
    type="password",
    help="Exemplo: postgresql://user:password@ep-cool-name.us-east-2.aws.neon.tech/neondb?sslmode=require"
)

# ---------------------------------------------------------
# Interface Gráfica - Painel Principal
# ---------------------------------------------------------
st.title("👁️ Sistema de Segmentação de Imagens e Metadados")
st.write("Faça upload de uma imagem, processe com IA leve (YOLOv8) e grave os dados diretamente na nuvem (Neon.tech).")

uploaded_file = st.file_uploader("Escolha uma imagem (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    file_size_kb = round(len(image_bytes) / 1024, 2)
    
    pil_image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(pil_image)
    
    height, width, channels = img_array.shape

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Imagem Original")
        st.image(pil_image, use_container_width=True)

    # Botão de Execução
    if st.button("🚀 Processar Imagem", type="primary"):
        if not db_url:
            st.error("❌ Por favor, forneça a String de Conexão do Neon.tech no menu lateral ou defina a variável DATABASE_URL no Render.")
        else:
            try:
                # 1. Inicializar tabela no Neon.tech se necessário
                init_db(db_url)

                # 2. Executar Segmentação com YOLOv8
                with st.spinner("Processando segmentação com visão computacional..."):
                    results = model(pil_image)
                    result = results[0]

                    annotated_frame = result.plot()
                    annotated_pil = Image.fromarray(annotated_frame[..., ::-1])

                with col2:
                    st.subheader("🎯 Imagem Segmentada")
                    st.image(annotated_pil, use_container_width=True)

                # 3. Processar Conteúdo Lido
                detected_counts = {}
                objects_summary = []

                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        cls_id = int(box.cls[0].item())
                        class_name = model.names[cls_id]
                        confidence = float(box.conf[0].item())

                        detected_counts[class_name] = detected_counts.get(class_name, 0) + 1
                        objects_summary.append({
                            "class": class_name,
                            "confidence": round(confidence, 2)
                        })

                total_objects = sum(detected_counts.values())

                # 4. Gravar Dados no Neon.tech
                save_analysis_to_db(
                    connection_string=db_url,
                    filename=uploaded_file.name,
                    width=width,
                    height=height,
                    channels=channels,
                    file_size_kb=file_size_kb,
                    detected_objects=objects_summary,
                    total_objects=total_objects
                )

                st.success("✅ Processamento concluído e dados salvos no Neon.tech com sucesso!")
                st.divider()

                # 5. Exibição detalhada dos resultados no Dashboard
                st.subheader("📋 O que está na imagem (Resultado da Leitura)")
                
                if total_objects > 0:
                    description_text = "A imagem contém: " + ", ".join([f"**{count} {name}(s)**" for name, count in detected_counts.items()])
                    st.markdown(f"### {description_text}")
                else:
                    st.warning("Nenhum objeto conhecido pelo modelo foi segmentado na imagem.")

                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Dimensões", f"{width} x {height} px")
                col_m2.metric("Canais de Cor", f"{channels} (RGB)")
                col_m3.metric("Tamanho do Arquivo", f"{file_size_kb} KB")
                col_m4.metric("Total de Objetos", total_objects)

                st.subheader("🔍 Detalhamento das Detecções")
                st.json({
                    "resumo_contagem": detected_counts,
                    "detalhes_objetos": objects_summary
                })

            except Exception as e:
                st.error(f"⚠️ Erro durante o processamento ou conexão com o banco de dados: {str(e)}")
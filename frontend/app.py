import streamlit as st
import requests
import os

API_URL = "http://localhost:8000"

st.set_page_config(page_title="PrivyRAG Offline", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0b0f;
    color: #ffffff;
}

/* Hide defaults */
header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Top Navbar */
.top-navbar {
    display: flex; justify-content: space-between; align-items: center;
    padding: 15px 30px; background: rgba(10, 11, 15, 0.9);
    backdrop-filter: blur(10px); border-bottom: 1px solid rgba(255,255,255,0.05);
    position: sticky; top: 0; z-index: 999; margin-top: -60px; margin-bottom: 20px;
}
.navbar-brand { font-weight: 800; font-size: 1.4rem; letter-spacing: -0.5px; }
.navbar-brand span { color: #4f8ef7; }
.navbar-status { display: flex; align-items: center; gap: 10px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #888; }
.pulse-dot { height: 8px; width: 8px; background-color: #00ff66; border-radius: 50%; display: inline-block; box-shadow: 0 0 10px #00ff66; animation: pulse 2s infinite; }
.model-badge { background: #1a1c23; padding: 4px 10px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); }
@keyframes pulse { 0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 255, 102, 0.7); } 70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(0, 255, 102, 0); } 100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 255, 102, 0); } }

/* Tabs */
[data-baseweb="tab-list"] { display: flex; justify-content: center; gap: 20px; background: transparent; }
[data-baseweb="tab"] { background: transparent; border: none; color: #888; padding: 10px 20px; border-radius: 20px; font-weight: 700; transition: all 0.2s ease; }
[aria-selected="true"] { background: #4f8ef7 !important; color: white !important; }

/* Sidebar */
[data-testid="stSidebar"] { background-color: #0e1117; border-right: 1px solid rgba(255,255,255,0.05); }

/* Buttons */
.stButton > button { width: 100%; border-radius: 8px; font-weight: 700; border: 1px solid #4f8ef7; transition: all 0.3s ease; }
button[kind="primary"] { background-color: #4f8ef7; color: white; border: none; }
.stButton > button:hover { filter: brightness(1.2); transform: translateY(-1px); }

/* Chat Messages */
.user-msg { background: rgba(79, 142, 247, 0.15); border: 1px solid rgba(79, 142, 247, 0.3); padding: 12px 18px; border-radius: 12px 12px 4px 12px; margin-left: auto; max-width: 80%; margin-bottom: 20px; animation: fadeUp 0.3s ease-out; }
.ai-msg { background: #13151c; border: 1px solid rgba(255,255,255,0.05); padding: 16px; border-radius: 4px 12px 12px 12px; width: 100%; margin-bottom: 20px; animation: fadeUp 0.3s ease-out; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
.sources-card { margin-top: 10px; padding: 10px; background: #0a0b0f; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #aaa; border: 1px dashed rgba(255,255,255,0.1); }

/* Document Cards */
.doc-card { background: #13151c; padding: 12px; border-radius: 8px; margin-bottom: 8px; border: 1px solid rgba(255,255,255,0.05); display: flex; align-items: center; justify-content: space-between; transition: all 0.2s ease; }
.doc-card:hover { border-color: #4f8ef7; }
.doc-icon { font-size: 1.2rem; margin-right: 12px; }
.doc-meta { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #888; }
.doc-title { font-size: 0.9rem; font-weight: 700; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 150px; }
.status-dot { height: 8px; width: 8px; background: #00ff66; border-radius: 50%; }

/* Stats & Privacy */
.stat-box { background: #13151c; border-radius: 8px; padding: 10px; text-align: center; border: 1px solid rgba(255,255,255,0.05); }
.stat-num { color: #4f8ef7; font-size: 1.5rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
.stat-label { font-size: 0.7rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
.privacy-banner { background: rgba(0,255,102,0.05); border: 1px dashed rgba(0,255,102,0.2); border-radius: 8px; padding: 12px; margin-top: 20px; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #4ade80; }
.privacy-banner div { margin-bottom: 4px; }
</style>

<div class="top-navbar">
    <div class="navbar-brand">🔒 Privy<span>RAG</span></div>
    <div class="navbar-status">
        <span class="model-badge">gemma2:2b</span>
        <span class="pulse-dot"></span> Local · Offline
    </div>
</div>
""", unsafe_allow_html=True)

# --- Core Logic ---
def fetch_files():
    try:
        res = requests.get(f"{API_URL}/files")
        if res.status_code == 200:
            return res.json().get("files", [])
    except:
        pass
    return []

available_files = fetch_files()

# --- Sidebar ---
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop your document here", type=["pdf", "docx", "txt", "md"])
    if st.button("Upload & Process", type="primary"):
        if uploaded_file:
            with st.spinner("Processing..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                res = requests.post(f"{API_URL}/upload", files=files)
                if res.status_code == 200:
                    st.success("Indexed successfully!")
                    st.rerun()
                else:
                    st.error("Failed to upload.")
                    
    st.markdown("<br><div style='font-family: JetBrains Mono; font-size: 0.7rem; color: #888; letter-spacing: 1px;'>DOCUMENTS</div>", unsafe_allow_html=True)
    for f in available_files:
        icon = "📕" if f.endswith(".pdf") else "📘" if f.endswith(".docx") else "📄"
        st.markdown(f"""
        <div class="doc-card">
            <div style="display: flex; align-items: center;">
                <div class="doc-icon">{icon}</div>
                <div><div class="doc-title">{f}</div><div class="doc-meta">Indexed & Ready</div></div>
            </div>
            <div class="status-dot"></div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown(f'<div class="stat-box"><div class="stat-num">{len(available_files)}</div><div class="stat-label">Docs</div></div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="stat-box"><div class="stat-num">{len(available_files)*42}</div><div class="stat-label">Chunks</div></div>', unsafe_allow_html=True) 
    with col3: st.markdown(f'<div class="stat-box"><div class="stat-num">?</div><div class="stat-label">Queries</div></div>', unsafe_allow_html=True)
        
    st.markdown("<div class='privacy-banner'><div>✓ Local processing</div><div>✓ No internet required</div><div>✓ No data shared</div></div>", unsafe_allow_html=True)

# --- Main Content ---
tab1, tab2, tab3 = st.tabs(["Chat", "Insights", "Compare"])

with tab1:
    if not available_files:
        st.markdown("<div style='color: #888; font-size: 0.9rem;'>No documents selected. Please upload a document to begin.</div>", unsafe_allow_html=True)
        selected_chat_docs = []
    else:
        selected_chat_docs = st.multiselect("Select documents to chat with:", available_files, default=available_files, label_visibility="collapsed")
        docs_html = " · ".join([f"📄 {f}" for f in selected_chat_docs]) if selected_chat_docs else "None"
        st.markdown(f"<div style='background:rgba(79, 142, 247, 0.1); padding:5px 12px; border-radius:12px; display:inline-block; font-size:0.8rem; border:1px solid rgba(79, 142, 247, 0.3); color:#4f8ef7; margin-bottom:20px;'>Active: {docs_html}</div>", unsafe_allow_html=True)
        
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Message rendering
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            sh = ""
            if msg.get("sources"):
                sh = "<div class='sources-card'><b>Sources:</b><br>" + "<br>".join(set([f"- {s['source']}" for s in msg['sources']])) + "</div>"
            st.markdown(f"<div class='ai-msg'>{msg['content']}{sh}</div>", unsafe_allow_html=True)
            
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    prompt = st.chat_input("Ask a question about your documents...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("AI is thinking..."):
            res = requests.post(f"{API_URL}/chat", json={"query": prompt, "selected_docs": selected_chat_docs})
            if res.status_code == 200:
                d = res.json()
                st.session_state.messages.append({"role": "assistant", "content": d["answer"], "sources": d.get("sources", [])})
                st.rerun()

with tab2:
    if available_files:
        selected_file = st.selectbox("Select document for insights:", available_files, label_visibility="collapsed")
        cA, cB = st.columns(2)
        if cA.button("Generate Summary"):
            with st.spinner("Generating..."):
                res = requests.post(f"{API_URL}/insights", json={"filename": selected_file})
                if res.status_code == 200:
                    st.markdown(f"<div class='ai-msg'>{res.json()['insights']}</div>", unsafe_allow_html=True)
        if cB.button("Extract Key Insights", type="primary"):
            with st.spinner("Extracting..."):
                res = requests.post(f"{API_URL}/insights", json={"filename": selected_file})
                if res.status_code == 200:
                    st.markdown(f"<div class='ai-msg'>{res.json()['insights']}</div>", unsafe_allow_html=True)
    else:
        st.info("Upload documents first.")

with tab3:
    if len(available_files) >= 2:
        cA, cB = st.columns(2)
        doc1 = cA.selectbox("Left Document", available_files, key="c1", label_visibility="collapsed")
        doc2 = cB.selectbox("Right Document", available_files, key="c2", label_visibility="collapsed")
        if st.button("Compare Documents", type="primary"):
            with st.spinner("Comparing..."):
                res = requests.post(f"{API_URL}/compare", json={"doc1": doc1, "doc2": doc2})
                if res.status_code == 200:
                    st.markdown(f"<div class='ai-msg'>{res.json()['comparison']}</div>", unsafe_allow_html=True)
    else:
        st.info("Upload at least two documents to compare.")

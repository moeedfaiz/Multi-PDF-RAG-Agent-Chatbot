import os, json, requests, html
import streamlit as st
from datetime import datetime
from typing import List, Dict, Any

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
DEFAULT_API_KEY = os.getenv("DEFAULT_API_KEY", "dev-key")


# ---------------------- Page / Theme ----------------------
st.set_page_config(page_title="PDF RAG Chat", page_icon="📄", layout="wide")

st.markdown(
    """
<style>
/* ---- Global ---- */
.block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; max-width: 1200px; }
.small-muted { color: rgba(255,255,255,0.72); font-size: 0.92rem; }
hr { margin: 0.75rem 0 1rem 0; opacity: 0.22; }

/* ---- Hero ---- */
.hero {
  padding: 18px 18px 14px 18px;
  border-radius: 18px;
  background: radial-gradient(1200px 220px at 0% 0%, rgba(99,102,241,0.35), transparent 60%),
              radial-gradient(1000px 220px at 100% 0%, rgba(14,165,233,0.30), transparent 60%),
              linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
  border: 1px solid rgba(255,255,255,0.10);
}
.hero h1 { margin: 0; font-size: 1.65rem; letter-spacing: 0.2px; }
.hero p { margin: 6px 0 0 0; opacity: 0.90; }
.chips { margin-top: 10px; display:flex; gap: 8px; flex-wrap: wrap; }
.chip {
  display: inline-flex; align-items:center; gap: 6px;
  padding: 4px 10px; border-radius: 999px; font-size: 0.82rem;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.03);
}

/* ---- Cards ---- */
.card {
  padding: 14px;
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
}
.card-title { font-weight: 800; margin-bottom: 10px; letter-spacing: 0.2px; }

/* ---- Chat bubbles ---- */
.bubble-q {
  padding: 12px 14px;
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(99,102,241,0.15);
  margin-bottom: 10px;
}
.bubble-a {
  padding: 12px 14px;
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.035);
  line-height: 1.6;
  overflow-wrap: anywhere;
}

/* ---- Badges ---- */
.badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 0.80rem;
  border: 1px solid rgba(255,255,255,0.16);
  margin-right: 8px;
}
.badge-ok { background: rgba(34,197,94,0.14); }
.badge-warn { background: rgba(234,179,8,0.14); }
.badge-muted { background: rgba(148,163,184,0.10); }

/* ---- Sidebar spacing ---- */
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
.sidebar-card {
  padding: 12px; border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.02);
  margin-bottom: 12px;
}
.sidebar-card h3 { margin: 0 0 8px 0; font-size: 1.02rem; }

/* ---- Small status pill ---- */
.pill-ok {
  display:inline-block; padding:2px 10px; border-radius:999px;
  background: rgba(34,197,94,0.18); border:1px solid rgba(34,197,94,0.25);
}
.pill-bad {
  display:inline-block; padding:2px 10px; border-radius:999px;
  background: rgba(239,68,68,0.15); border:1px solid rgba(239,68,68,0.25);
}

/* ---- Divider ---- */
.chat-divider {
  height: 1px;
  margin: 12px 0;
  background: rgba(255,255,255,0.10);
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------- Helpers ----------------------
def safe_headers(api_key: str) -> Dict[str, str]:
    return {"X-API-Key": api_key} if api_key else {}


def backend_health() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=4)
        return r.status_code == 200
    except Exception:
        return False


@st.cache_data(ttl=8)
def cached_get_docs(api_key: str, backend_url: str) -> List[Dict[str, Any]]:
    if not api_key:
        return []
    r = requests.get(f"{backend_url}/documents", headers=safe_headers(api_key), timeout=30)
    r.raise_for_status()
    return r.json().get("docs", [])


def doc_label(d: dict) -> str:
    fid = d.get("file_id", "")
    fn = d.get("filename", "(no name)")
    chunks = d.get("num_chunks", 0)
    ingested = bool(d.get("ingested", False))
    badge = "✅ ingested" if ingested else "⏳ not ingested"
    return f"{fn} • {badge} • chunks={chunks} • {fid[:8]}"


def pretty_badge(ingested: bool) -> str:
    if ingested:
        return '<span class="badge badge-ok">✅ ingested</span>'
    return '<span class="badge badge-warn">⏳ not ingested</span>'


def fmt_dt(s: str) -> str:
    if not s:
        return ""
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return s


def fetch_docs_once(api_key: str) -> List[Dict[str, Any]]:
    try:
        return cached_get_docs(api_key, BACKEND_URL)
    except Exception:
        return []


def render_answer_html(text: str) -> str:
    safe = html.escape(text or "").replace("\n", "<br/>")
    return f'<div class="bubble-a">{safe}</div>'


# ---------------------- State ----------------------
if "refresh_key" not in st.session_state:
    st.session_state.refresh_key = 0
if "chat_history" not in st.session_state:
    # item: {"q":..., "a":..., "citations":[...], "provider":"", "model":""}
    st.session_state.chat_history = []
if "last_selected_docs" not in st.session_state:
    st.session_state.last_selected_docs = []
if "last_llm_provider" not in st.session_state:
    st.session_state.last_llm_provider = ""
if "last_llm_model" not in st.session_state:
    st.session_state.last_llm_model = ""


# ---------------------- Hero ----------------------
ok = backend_health()
provider_chip = st.session_state.last_llm_provider or "unknown"
model_chip = st.session_state.last_llm_model or "unknown"

st.markdown(
    f"""
<div class="hero">
  <h1>📄 PDF RAG Chat</h1>
  <p class="small-muted">
    Ask questions across one or more PDFs. Stream answers with citations.
    <span style="margin-left:10px;" class="{"pill-ok" if ok else "pill-bad"}">
      Backend: {"OK ✅" if ok else "Offline ❌"}
    </span>
  </p>
  <div class="chips">
    <span class="chip">🔎 RAG: Qdrant</span>
    <span class="chip">🧠 LLM: {html.escape(provider_chip)} / {html.escape(model_chip)}</span>
    <span class="chip">🧾 Citations: page + snippet</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")


# ---------------------- Sidebar ----------------------
st.sidebar.markdown("## Control Panel")

api_key = st.sidebar.text_input("X-API-Key", value=DEFAULT_API_KEY, type="password")
headers = safe_headers(api_key)

cA, cB = st.sidebar.columns(2)
if cA.button("🔄 Refresh", use_container_width=True, disabled=not api_key):
    cached_get_docs.clear()
    st.session_state.refresh_key += 1

with cB:
    if ok:
        st.success("Backend OK")
    else:
        st.error("Offline")

st.sidebar.markdown("---")


# ---------------------- Upload (Sidebar) ----------------------
st.sidebar.markdown(
    """
<div class="sidebar-card">
  <h3>1) Upload</h3>
</div>
""",
    unsafe_allow_html=True,
)

uploaded = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])
auto_ingest = st.sidebar.toggle("Auto-ingest after upload", value=True)

if st.sidebar.button("⬆️ Upload PDF", disabled=uploaded is None or not api_key, use_container_width=True):
    files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
    r = requests.post(
        f"{BACKEND_URL}/upload",
        params={"ingest": str(auto_ingest).lower()},
        files=files,
        headers=headers,
        timeout=900,
    )
    if r.status_code != 200:
        st.sidebar.error(r.text)
        st.stop()

    st.sidebar.success("Uploaded ✅")
    cached_get_docs.clear()
    st.session_state.refresh_key += 1

st.sidebar.markdown("---")


# ---------------------- Manage (Sidebar) ----------------------
st.sidebar.markdown(
    """
<div class="sidebar-card">
  <h3>2) Manage PDFs</h3>
</div>
""",
    unsafe_allow_html=True,
)

_ = st.session_state.refresh_key
docs = fetch_docs_once(api_key)

label_to_id = {doc_label(d): d["file_id"] for d in docs}
selected_manage = st.sidebar.selectbox("Select PDF", [""] + list(label_to_id.keys()))
manage_file_id = label_to_id.get(selected_manage)

m1, m2 = st.sidebar.columns(2)

if m1.button("📥 Ingest", disabled=not manage_file_id or not api_key, use_container_width=True):
    r = requests.post(f"{BACKEND_URL}/ingest/{manage_file_id}", headers=headers, timeout=1200)
    if r.status_code != 200:
        st.sidebar.error(r.text)
        st.stop()
    st.sidebar.success("Ingested ✅")
    cached_get_docs.clear()
    st.session_state.refresh_key += 1

confirm = st.sidebar.checkbox("Confirm delete", value=False)
if m2.button("🗑️ Delete", disabled=not manage_file_id or not api_key or not confirm, use_container_width=True):
    r = requests.delete(f"{BACKEND_URL}/documents/{manage_file_id}", headers=headers, timeout=240)
    if r.status_code != 200:
        st.sidebar.error(r.text)
        st.stop()
    st.sidebar.success("Deleted ✅")
    cached_get_docs.clear()
    st.session_state.refresh_key += 1

st.sidebar.caption("Tip: Leave Ask PDFs empty to search across all ingested PDFs.")
st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear chat history", use_container_width=True):
    st.session_state.chat_history = []


# ---------------------- Main Layout ----------------------
left, right = st.columns([1.35, 0.95], gap="large")


# ---------------------- Ask (Main Left) ----------------------
with left:
    st.markdown('<div class="card"><div class="card-title">3) Ask</div>', unsafe_allow_html=True)

    docs = fetch_docs_once(api_key)
    label_to_id = {doc_label(d): d["file_id"] for d in docs}

    chosen = st.multiselect(
        "Choose PDFs (optional)",
        list(label_to_id.keys()),
        default=st.session_state.last_selected_docs,
        help="If empty, search across all ingested PDFs.",
    )
    st.session_state.last_selected_docs = chosen

    file_ids = [label_to_id[c] for c in chosen] if chosen else None

    q = st.text_area("Question", placeholder="Ask something from your PDF…", height=90)

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    top_k = c1.slider("Top-K", 3, 20, 8)
    max_tokens = c2.slider("Max tokens", 64, 2048, 1024, step=64)  # ✅ bigger default
    use_stream = c3.toggle("Stream", value=True)
    show_citations = c4.toggle("Show citations", value=True)

    ask = st.button("✨ Ask", disabled=not q.strip() or not api_key, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

    # ---------- Chat history ----------
    if st.session_state.chat_history:
        with st.expander("💬 Conversation", expanded=True):
            for item in st.session_state.chat_history[-10:]:
                st.markdown(
                    f'<div class="bubble-q"><b>Q:</b> {html.escape(item.get("q",""))}</div>',
                    unsafe_allow_html=True,
                )

                tab_a, tab_c = st.tabs(["Response", "📌 Citations"])
                with tab_a:
                    # Show provider/model if present
                    p = item.get("provider") or ""
                    m = item.get("model") or ""
                    if p or m:
                        st.caption(f"LLM: {p} / {m}")
                    st.markdown(render_answer_html(item.get("a", "")), unsafe_allow_html=True)

                with tab_c:
                    if show_citations and item.get("citations"):
                        for c in item["citations"][:10]:
                            st.markdown(f"- **{c['source']}** (page {c['page']}) — {c['snippet']}")
                    else:
                        st.caption("No citations (or citations hidden).")

                st.markdown('<div class="chat-divider"></div>', unsafe_allow_html=True)

            st.caption("Showing last 10 messages.")

    if ask:
        payload = {"file_ids": file_ids, "question": q.strip(), "top_k": top_k, "max_tokens": max_tokens}

        st.markdown(
            f'<div class="bubble-q"><b>Q:</b> {html.escape(q.strip())}</div>',
            unsafe_allow_html=True,
        )

        # ---------------- Streaming (UPDATED: handles `final`) ----------------
        if use_stream:
            resp = requests.post(
                f"{BACKEND_URL}/chat/stream",
                json=payload,
                headers={**headers, "Accept": "text/event-stream"},
                stream=True,
                timeout=1200,
            )
            if resp.status_code != 200:
                st.error(resp.text)
                st.stop()

            box = st.empty()
            answer = ""
            citations: List[Dict[str, Any]] = []
            provider = ""
            model = ""

            try:
                with st.spinner("Streaming…"):
                    for raw in resp.iter_lines(decode_unicode=True):
                        if not raw or not raw.startswith("data: "):
                            continue

                        msg = json.loads(raw.replace("data: ", ""))

                        if msg.get("type") == "meta":
                            citations = msg.get("citations", []) or []
                            provider = msg.get("provider", "") or ""
                            model = msg.get("model", "") or ""

                        elif msg.get("type") == "refused":
                            # backend may also send final after refused
                            warn = (msg.get("answer") or "Refused").strip()
                            st.warning(warn)

                        elif msg.get("type") == "token":
                            answer += msg.get("token", "")
                            box.markdown(render_answer_html(answer + "▌"), unsafe_allow_html=True)

                        elif msg.get("type") == "final":
                            # ✅ ensure we display the FULL final answer
                            answer = (msg.get("answer") or "").strip()
                            box.markdown(render_answer_html(answer), unsafe_allow_html=True)

                        elif msg.get("type") == "done":
                            break

            except requests.exceptions.ChunkedEncodingError:
                st.error("Streaming disconnected (backend restarted). Try again.")
            except Exception as e:
                st.error(f"Streaming error: {e}")

            # Persist last used model in hero chip
            st.session_state.last_llm_provider = provider or st.session_state.last_llm_provider
            st.session_state.last_llm_model = model or st.session_state.last_llm_model

            # Save history
            st.session_state.chat_history.append(
                {"q": q.strip(), "a": answer.strip(), "citations": citations, "provider": provider, "model": model}
            )

            if show_citations and citations:
                with st.expander("📌 Citations", expanded=False):
                    for c in citations[:12]:
                        st.markdown(f"- **{c['source']}** (page {c['page']}) — {c['snippet']}")

        # ---------------- Non-streaming ----------------
        else:
            with st.spinner("Thinking…"):
                r = requests.post(f"{BACKEND_URL}/chat", json=payload, headers=headers, timeout=1200)

            if r.status_code != 200:
                st.error(r.text)
                st.stop()

            data = r.json()
            ans = (data.get("answer", "") or "").strip()
            citations = data.get("citations", []) or []

            st.markdown(render_answer_html(ans), unsafe_allow_html=True)

            st.session_state.chat_history.append({"q": q.strip(), "a": ans, "citations": citations})

            if show_citations and citations:
                with st.expander("📌 Citations", expanded=False):
                    for c in citations[:12]:
                        st.markdown(f"- **{c['source']}** (page {c['page']}) — {c['snippet']}")


# ---------------------- Library (Main Right) ----------------------
with right:
    st.markdown('<div class="card"><div class="card-title">📚 Library</div>', unsafe_allow_html=True)

    docs = fetch_docs_once(api_key)

    if not api_key:
        st.info("Enter API key to load your PDFs.")
    elif not docs:
        st.info("No PDFs yet. Upload one from the sidebar.")
    else:
        qlib = st.text_input("Search library", placeholder="type filename or file_id…")
        sort_by = st.selectbox("Sort by", ["Newest", "Most chunks", "Filename A→Z"])

        def key_newest(d):
            return d.get("created_at") or ""

        def key_chunks(d):
            return int(d.get("num_chunks") or 0)

        filtered = docs
        if qlib.strip():
            qq = qlib.lower().strip()
            filtered = [
                d for d in docs
                if qq in (d.get("filename", "") or "").lower()
                or qq in (d.get("file_id", "") or "").lower()
            ]

        if sort_by == "Newest":
            filtered = sorted(filtered, key=key_newest, reverse=True)
        elif sort_by == "Most chunks":
            filtered = sorted(filtered, key=key_chunks, reverse=True)
        else:
            filtered = sorted(filtered, key=lambda d: (d.get("filename") or "").lower())

        ing = sum(1 for d in docs if d.get("ingested"))
        total = len(docs)
        st.markdown(
            f"""
<div class="chips" style="margin-top:0;">
  <span class="chip">📄 {total} PDFs</span>
  <span class="chip">✅ {ing} ingested</span>
  <span class="chip">🧩 {sum(int(d.get("num_chunks") or 0) for d in docs)} chunks</span>
</div>
""",
            unsafe_allow_html=True,
        )
        st.write("")

        for d in filtered[:40]:
            fid = d.get("file_id", "")
            fn = d.get("filename", "(no name)")
            chunks = d.get("num_chunks", 0)
            ingested = bool(d.get("ingested", False))
            created = fmt_dt(d.get("created_at"))

            st.markdown(
                f"""
<div style="padding:12px 12px; border-radius:16px; border:1px solid rgba(255,255,255,0.10);
            background: rgba(255,255,255,0.02); margin-bottom:10px;">
  <div style="font-weight:800; font-size:0.98rem;">{html.escape(fn)}</div>
  <div style="margin-top:7px;">
    {pretty_badge(ingested)}
    <span class="badge badge-muted">chunks: {chunks}</span>
    <span class="badge badge-muted">{fid[:8]}</span>
    {"<span class='badge badge-muted'>"+html.escape(created)+"</span>" if created else ""}
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

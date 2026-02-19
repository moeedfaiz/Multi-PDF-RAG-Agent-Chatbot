import logging
from fastapi import APIRouter, Depends

from ...deps import get_tenant_id
from ...schemas.chat import ChatRequest, ChatResponse, Citation
from ...services.rag import retrieve, build_context, make_citations
from ...services.guardrails import should_refuse
from ...services.llm import llm_generate
from ...services.timing import T
from ...config import settings

router = APIRouter()
log = logging.getLogger("timing")

SYSTEM = """You are a PDF question-answering assistant.

Hard rules:
- Use ONLY the provided CONTEXT. Do NOT use filenames or outside knowledge.
- If the answer is not present in the context, say exactly:
  "I don't have enough information in the uploaded document(s) to answer that."
- Never respond with 1 vague sentence. Be detailed and specific.

Answer style:
- Default: 1 paragraph, 6–9 sentences.
- For questions like "tell me about this pdf", "what is in this pdf", "summarize", "brief", "overview":
  produce a structured brief with these sections (even if some are missing):
  1) What this document is
  2) Key entities (people/companies/IDs/dates)
  3) Main topics / sections covered
  4) Notable numbers / metrics (if any)
  5) What is NOT mentioned / unclear (if relevant)
- If the user asks for bullets/list: use bullet points.

All claims must be supported by the context.
Output must be plain text.
"""


def is_summary_question(q: str) -> bool:
    q = (q or "").lower()
    keys = [
        "tell me about",
        "what is in",
        "what information",
        "summarize",
        "summary",
        "brief",
        "overview",
        "describe this pdf",
        "what does this pdf contain",
    ]
    return any(k in q for k in keys)


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, tenant_id: str = Depends(get_tenant_id)):
    t = T("chat")

    summary_mode = is_summary_question(req.question)
    top_k = max(req.top_k, 16) if summary_mode else req.top_k

    log.info(
        "REQ /chat provider=%s q_len=%s top_k=%s max_tokens=%s file_ids=%s summary_mode=%s",
        settings.llm_provider,
        len(req.question),
        top_k,
        req.max_tokens,
        req.file_ids,
        summary_mode,
    )

    pairs = retrieve(
        req.question,
        top_k=top_k,
        file_ids=req.file_ids,
        tenant_id=tenant_id,
    )
    t.mark(f"retrieve pairs={len(pairs)}")

    docs = [p[0] for p in pairs]
    scores = [p[1] for p in pairs]

    if should_refuse(docs):
        t.mark("refuse_check")
        return ChatResponse(
            answer="I don't have enough information in the uploaded document(s) to answer that.",
            refused=True,
            citations=[],
        )

    context = build_context(docs)
    t.mark("build_context")

    extra = ""
    if summary_mode:
        extra = "\nIMPORTANT: Write a structured brief with the 5 sections listed in the instructions."

    prompt = f"""{SYSTEM}

CONTEXT:
{context}

QUESTION:
{req.question}
{extra}

ANSWER:"""

    answer = (llm_generate(prompt, max_tokens=req.max_tokens) or "").strip()
    t.mark("llm_generate")

    # Retry once if too short (prevents 1-liners)
    if len(answer) < 120 and context.strip():
        retry_prompt = (
            prompt
            + "\n\nIMPORTANT: Expand the answer. Minimum 6 sentences OR 8 bullet points. Be specific."
        )
        answer = (llm_generate(retry_prompt, max_tokens=req.max_tokens) or "").strip()
        t.mark("llm_generate_retry")

    citations = [Citation(**c) for c in make_citations(docs, scores)]
    t.mark("make_citations")

    log.info("DONE /chat")
    return ChatResponse(answer=answer, refused=False, citations=citations)

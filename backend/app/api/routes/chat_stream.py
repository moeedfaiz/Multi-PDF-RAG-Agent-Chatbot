import json
import logging
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ...deps import get_tenant_id
from ...schemas.chat import ChatRequest
from ...services.rag import retrieve, build_context, make_citations
from ...services.guardrails import should_refuse
from ...services.llm import llm_stream
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
  produce a structured brief with these sections:
  1) What this document is
  2) Key entities (people/companies/IDs/dates)
  3) Main topics / sections covered
  4) Notable numbers / metrics (if any)
  5) What is NOT mentioned / unclear (if relevant)
- If the user asks for bullets/list: use bullet points.

All claims must be supported by the context.
Output must be plain text.
"""


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


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
        "what is this doc about",
    ]
    return any(k in q for k in keys)


@router.post("/chat/stream")
def chat_stream(req: ChatRequest, tenant_id: str = Depends(get_tenant_id)):
    t = T("chat_stream")

    summary_mode = is_summary_question(req.question)
    top_k = max(req.top_k, 16) if summary_mode else req.top_k

    log.info(
        "REQ /chat/stream provider=%s model=%s q_len=%s top_k=%s max_tokens=%s file_ids=%s summary_mode=%s",
        settings.llm_provider,
        getattr(settings, "gemini_model", None) or getattr(settings, "ollama_model", None),
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

    citations = make_citations(docs, scores)
    t.mark("make_citations")

    def gen():
        # Send provider/model info too (frontend can show it if you want)
        yield sse(
            {
                "type": "meta",
                "citations": citations,
                "provider": settings.llm_provider,
                "model": getattr(settings, "gemini_model", None) or getattr(settings, "ollama_model", None),
            }
        )
        t.mark("sent_meta")

        if should_refuse(docs):
            msg = "I don't have enough information in the uploaded document(s) to answer that."
            yield sse({"type": "refused", "answer": msg})
            yield sse({"type": "final", "answer": msg})
            yield sse({"type": "done"})
            t.mark("refused_done")
            return

        context = build_context(docs)
        t.mark("build_context")

        extra = ""
        if summary_mode:
            extra = "\nIMPORTANT: Write the structured brief with 5 sections. Be detailed."

        prompt = f"""{SYSTEM}

CONTEXT:
{context}

QUESTION:
{req.question}
{extra}

ANSWER:"""

        t.mark("llm_call_start")

        try:
            first = True
            emitted_any = False
            full_answer = ""

            for token in llm_stream(prompt, max_tokens=req.max_tokens):
                if first:
                    t.mark("llm_first_token")
                    first = False
                emitted_any = True
                full_answer += token
                yield sse({"type": "token", "token": token})

            if not emitted_any:
                full_answer = "I don't have enough information in the uploaded document(s) to answer that."
                yield sse({"type": "token", "token": full_answer})

            # Retry once if too short (prevents one-liners)
            if len(full_answer.strip()) < 160 and context.strip():
                retry_prompt = (
                    prompt
                    + "\n\nIMPORTANT: Expand. Minimum 6–9 sentences OR 10 bullet points. Be specific and grounded."
                )
                full_answer = ""
                for token in llm_stream(retry_prompt, max_tokens=req.max_tokens):
                    full_answer += token

            # Always send final full answer
            yield sse({"type": "final", "answer": full_answer.strip()})

            t.mark("llm_stream_done")
            yield sse({"type": "done"})

        except Exception as e:
            log.exception("LLM stream crashed")
            msg = f"LLM error: {str(e)}"
            yield sse({"type": "refused", "answer": msg})
            yield sse({"type": "final", "answer": msg})
            yield sse({"type": "done"})

    return StreamingResponse(gen(), media_type="text/event-stream")

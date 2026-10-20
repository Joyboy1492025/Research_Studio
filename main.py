"""
Research Studio: a web search feeding five agents (phidata + Ollama), shown as a streaming chat.

Setup:
    pip install -U ddgs phidata ollama gradio
    ollama pull llama3.2:3b
    python research_studio.py

Optional: set SEARCH_PROXY (for example http://user:pass@host:port) if your IP gets rate-limited.
"""
import os
import random
import re
import tempfile
import textwrap
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import gradio as gr
from phi.agent import Agent
from phi.model.ollama import Ollama
from phi.workflow import RunResponse, Workflow

DEFAULT_MODEL = "llama3.2:3b"
MODELS = ["llama3.2:3b", "llama3.1:8b", "qwen2.5:7b", "mistral:7b"]
PUSH_INTERVAL = 0.07  # seconds between UI refreshes while text streams

# Search resilience
RETRY_DELAYS = (3, 8, 15)  # seconds to wait after each failed attempt
MIN_GAP = 2.0  # minimum seconds between two requests to the search engine
CACHE_TTL = 600  # reuse results for the same topic for 10 minutes

MODE_RESEARCH = "Research a topic"
MODE_ASK = "Ask about the report"


# ---------------------------------------------------------------------------
# Web search: one request at a time, retried with backoff, cached
# ---------------------------------------------------------------------------
class SearchUnavailable(Exception):
    pass


_cache: dict = {}
_lock = threading.Lock()
_last_call = 0.0


def _client():
    try:
        from ddgs import DDGS  # maintained package: queries several engines
    except ImportError:
        from duckduckgo_search import DDGS  # old package, rate-limits much more often
    return DDGS(proxy=os.getenv("SEARCH_PROXY") or None, timeout=15)


def _search_once(topic: str, k: int) -> List[dict]:
    client, items, errors = _client(), [], []
    for kind in ("news", "text"):  # news is fresher; fall back to general results
        try:
            rows = getattr(client, kind)(topic, max_results=k) or []
        except Exception as exc:
            errors.append(exc)
            continue
        for r in rows:
            url = r.get("url") or r.get("href") or ""
            if url and all(url != i["url"] for i in items):
                items.append(
                    {
                        "title": (r.get("title") or url).strip(),
                        "url": url,
                        "snippet": (r.get("body") or "").strip(),
                        "date": (r.get("date") or "")[:10],
                        "source": r.get("source") or "",
                    }
                )
        if len(items) >= k:
            break
    if not items and errors:
        raise errors[-1]
    return items[:k]


def _throttle() -> None:
    global _last_call
    with _lock:
        wait = MIN_GAP - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


def search_with_retry(topic: str, k: int):
    """Generator: yields progress notices for stage 1, returns the list of results."""
    key = (topic.lower(), k)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]

    last: Optional[Exception] = None
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            _throttle()
            results = _search_once(topic, k)
            if results:
                _cache[key] = (time.time(), results)
                return results
            last = RuntimeError("The search returned no results.")
        except Exception as exc:
            last = exc
        if attempt < len(RETRY_DELAYS):
            wait = RETRY_DELAYS[attempt] + random.random()
            yield RunResponse(
                content=f"Search was refused or returned nothing (attempt {attempt + 1} of "
                f"{len(RETRY_DELAYS) + 1}). Retrying in {wait:.0f}s…",
                event="stage_set:1",
            )
            time.sleep(wait)

    raise SearchUnavailable(
        "Web search is rate-limited right now, so no sources could be fetched after several attempts.\n\n"
        "Wait a minute and try again, turn off **Search the web** in Settings to continue without sources, "
        "or set the `SEARCH_PROXY` environment variable. Also make sure you use the maintained package "
        "(`pip install -U ddgs`).\n\n"
        f"Last error: `{type(last).__name__}: {last}`"
    )


def results_md(results: List[dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        title = r["title"].replace("[", "(").replace("]", ")")
        meta = ", ".join(x for x in (r["source"], r["date"]) if x)
        lines.append(f"{i}. [{title}]({r['url']})" + (f" ({meta})" if meta else ""))
    return "\n".join(lines)


def results_text(results: List[dict]) -> str:
    return "\n\n".join(
        f"{i}. {r['title']}\n   url: {r['url']}\n   date: {r['date'] or 'unknown'}\n   snippet: {r['snippet']}"
        for i, r in enumerate(results, 1)
    )


# ---------------------------------------------------------------------------
# Pipeline definition: one list drives the agents and the UI tracker
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Stage:
    name: str
    role: str
    instructions: Optional[List[str]] = None  # None = not an agent (the web search step)


STAGES = [
    Stage("Web search", "Finds sources on the web"),
    Stage(
        "Research Collector",
        "Picks the relevant sources",
        [
            "Use only the search results provided.",
            "Select the top 5 relevant sources.",
            "Return title, summary, publication date and source url.",
        ],
    ),
    Stage(
        "Source Validator",
        "Drops unreliable sources",
        [
            "Check source credibility.",
            "Remove unreliable sources.",
            "Keep only high-quality sources.",
        ],
    ),
    Stage(
        "Deduplication Expert",
        "Merges repeated stories",
        [
            "Detect duplicate articles.",
            "Merge repeated news.",
            "Keep only unique information.",
        ],
    ),
    Stage(
        "Research Analyst",
        "Finds trends and insights",
        [
            "Identify trends.",
            "Compare sources.",
            "Analyze impact.",
            "Extract key insights.",
        ],
    ),
    Stage(
        "Senior Journalist",
        "Writes the report",
        [
            "Write a professional report.",
            "Use NYT-style journalism.",
            "Create headings and subheadings.",
            "Include executive summary.",
            "Include references.",
            "Use markdown formatting.",
            "Only use facts from the research provided. Never invent sources or URLs.",
        ],
    ),
]

REPORT_TEMPLATE = textwrap.dedent(
    """\
    Topic: {topic}

    Unique Research:
    {research}

    Analysis:
    {analysis}

    Create a professional report with:

    1. Executive Summary
    2. Latest Developments
    3. Key Insights
    4. Industry Impact
    5. Future Predictions
    6. References
    """
)


def build_agents(model_id: str) -> List[Agent]:
    # No tools on purpose: search runs in plain Python (reliable), and agents without
    # tools stream their replies token by token even on older Ollama versions.
    return [
        Agent(name=s.name, model=Ollama(id=model_id), instructions=s.instructions, markdown=True)
        for s in STAGES
        if s.instructions
    ]


def collector_prompt(topic: str, results: List[dict]) -> str:
    if not results:
        return (
            f"Topic: {topic}\n\nNo live search results are available. Using only your own knowledge, "
            "list the most relevant developments you know of. State clearly that they are unverified "
            "and may be out of date."
        )
    return (
        f"Topic: {topic}\n\nSearch results:\n{results_text(results)}\n\n"
        "Use only these results. For each relevant source return the title, a short summary, "
        "the publication date and the source url."
    )


# ---------------------------------------------------------------------------
# Workflow. Progress is reported through RunResponse.event:
#   stage_start:N / stage_chunk:N (append text) / stage_set:N (replace text) / stage_end:N
# ---------------------------------------------------------------------------
def search_step(topic: str, k: int, use_web: bool):
    yield RunResponse(content="", event="stage_start:1")
    results: List[dict] = []
    if use_web:
        results = yield from search_with_retry(topic, k)
        yield RunResponse(content=results_md(results), event="stage_set:1")
    else:
        yield RunResponse(
            content="Web search is turned off. The agents rely on the model's own knowledge, "
            "so details may be out of date.",
            event="stage_set:1",
        )
    yield RunResponse(content="", event="stage_end:1")
    return results


def stream_stage(n: int, agent: Agent, prompt: str):
    """Run one agent, yield its tokens as they arrive, return the full text."""
    yield RunResponse(content="", event=f"stage_start:{n}")
    parts: List[str] = []
    for chunk in agent.run(prompt, stream=True):
        text = chunk.content
        if isinstance(text, str) and text:
            parts.append(text)
            yield RunResponse(content=text, event=f"stage_chunk:{n}")
    yield RunResponse(content="", event=f"stage_end:{n}")
    return "".join(parts)


class GenerateNewsReport(Workflow):
    def run(
        self, topic: str, model_id: str = DEFAULT_MODEL, max_results: int = 6, use_web: bool = True
    ) -> Iterator[RunResponse]:
        collector, validator, deduper, analyst, writer = build_agents(model_id)

        results = yield from search_step(topic, max_results, use_web)
        research = yield from stream_stage(2, collector, collector_prompt(topic, results))
        validated = yield from stream_stage(3, validator, f"Validate these sources:\n{research}")
        unique = yield from stream_stage(4, deduper, f"Remove duplicates:\n{validated}")
        insights = yield from stream_stage(5, analyst, f"Analyze:\n{unique}")
        yield from stream_stage(
            6,
            writer,
            REPORT_TEMPLATE.format(topic=topic, research=unique, analysis=insights),
        )


# ---------------------------------------------------------------------------
# Chat state and rendering
# ---------------------------------------------------------------------------
STATUS_LABEL = {"running": "Working…", "done": "Done", "error": "Failed"}
LAST = len(STAGES) - 1  # index of the writer; its text is the final chat message


@dataclass
class RunState:
    status: List[str] = field(default_factory=lambda: ["idle"] * len(STAGES))
    started: List[float] = field(default_factory=lambda: [0.0] * len(STAGES))
    elapsed: List[float] = field(default_factory=lambda: [0.0] * len(STAGES))
    text: List[str] = field(default_factory=lambda: [""] * len(STAGES))
    error: str = ""

    def pipeline_html(self) -> str:
        rows = []
        for i, stage in enumerate(STAGES):
            s = self.status[i]
            secs = time.time() - self.started[i] if s == "running" else self.elapsed[i]
            clock = f"{secs:.1f}s" if s != "idle" else ""
            glyph = {"done": "✓", "error": "!"}.get(s, str(i + 1))
            sub = stage.role if s == "idle" else STATUS_LABEL[s]
            rows.append(
                f'<div class="rs-step {s}"><div class="rs-node">{glyph}</div>'
                f'<div class="rs-meta"><b>{stage.name}</b><span>{sub}</span></div>'
                f'<span class="rs-time">{clock}</span></div>'
            )
        return '<div class="rs-pipe"><div class="rs-ptitle">Pipeline</div>' + "".join(rows) + "</div>"

    def messages(self) -> List[dict]:
        """Chat messages for this run: one collapsible bubble per step, then the report."""
        out: List[dict] = []
        for i, stage in enumerate(STAGES[:LAST]):
            s = self.status[i]
            if s == "idle":
                continue
            title = f"{i + 1}. {stage.name}"
            if s == "done":
                title += f" ({self.elapsed[i]:.1f}s)"
            elif s == "error":
                title += " (failed)"
            body = self.text[i].strip() or "_Working…_"
            if s == "running" and self.text[i].strip():
                body += " ▌"
            out.append(
                {
                    "role": "assistant",
                    "content": body,
                    "metadata": {"title": title, "status": "pending" if s == "running" else "done"},
                }
            )
        if self.status[LAST] != "idle":
            draft = self.text[LAST].strip() or "_Writing…_"
            out.append({"role": "assistant", "content": draft + (" ▌" if self.status[LAST] == "running" else "")})
        if self.error:
            out.append({"role": "assistant", "content": f"**The run failed.**\n\n{self.error}"})
        return out


_KEEP = object()


def ui(history, *, pipeline=None, file=_KEEP, busy=True, clear=False, report=None):
    """Values for: chatbot, pipeline, download button, input box, send button, report state."""
    return (
        history,
        pipeline if pipeline is not None else gr.update(),
        gr.update() if file is _KEEP else gr.update(value=file, visible=bool(file)),
        gr.update(value="") if clear else gr.update(),
        gr.update(interactive=not busy),
        report if report is not None else gr.update(),
    )


def noop():
    return tuple(gr.update() for _ in range(6))


def save_report(topic: str, text: str) -> Optional[str]:
    if not text.strip():
        return None
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:40] or "report"
    path = Path(tempfile.gettempdir()) / f"report-{time.strftime('%Y%m%d-%H%M%S')}-{slug}.md"
    path.write_text(text, encoding="utf-8")
    return str(path)


def explain_error(exc: Exception, model_id: str) -> str:
    if isinstance(exc, SearchUnavailable):
        return str(exc)
    return (
        f"`{type(exc).__name__}: {exc}`\n\n"
        f"If this is a connection error, start Ollama (`ollama serve`) and make sure the model "
        f"is installed (`ollama pull {model_id}`)."
    )


def research(topic: str, base: list, model_id: str, k: int, use_web: bool):
    st = RunState()
    began = time.time()
    yield ui(base, pipeline=st.pipeline_html(), file=None, clear=True)

    last_push = 0.0
    try:
        for ev in GenerateNewsReport().run(topic=topic, model_id=model_id, max_results=int(k), use_web=use_web):
            kind, _, num = (ev.event or "").partition(":")
            if not num.isdigit():
                continue
            i = int(num) - 1
            force = False

            if kind == "stage_start":
                st.status[i], st.started[i], force = "running", time.time(), True
            elif kind == "stage_chunk":
                st.text[i] += ev.content
            elif kind == "stage_set":
                st.text[i], force = ev.content, True
            elif kind == "stage_end":
                st.status[i], st.elapsed[i], force = "done", time.time() - st.started[i], True

            now = time.monotonic()
            if force or now - last_push >= PUSH_INTERVAL:
                last_push = now
                yield ui(base + st.messages(), pipeline=st.pipeline_html())
    except Exception as exc:  # search blocked, Ollama down, model missing, ...
        for i, s in enumerate(st.status):
            if s == "running":
                st.status[i], st.elapsed[i] = "error", time.time() - st.started[i]
        st.error = explain_error(exc, model_id)
        yield ui(base + st.messages(), pipeline=st.pipeline_html(), busy=False)
        return

    gr.Info(f"Report generated in {time.time() - began:.0f}s.")
    yield ui(
        base + st.messages(),
        pipeline=st.pipeline_html(),
        file=save_report(topic, st.text[LAST]),
        busy=False,
        report=st.text[LAST],
    )


def follow_up(question: str, base: list, report: str, model_id: str):
    if not report.strip():
        gr.Warning("Generate a report first, then ask questions about it.")
        yield noop()
        return

    agent = Agent(
        name="Report Assistant",
        model=Ollama(id=model_id),
        instructions=[
            "Answer using only the report provided.",
            "If the report does not contain the answer, say so.",
            "Be concise and use markdown.",
        ],
        markdown=True,
    )
    reply, last_push = "", 0.0
    yield ui(base + [{"role": "assistant", "content": "_Thinking…_"}], clear=True)
    try:
        for chunk in agent.run(f"Report:\n{report}\n\nQuestion: {question}", stream=True):
            if isinstance(chunk.content, str) and chunk.content:
                reply += chunk.content
                if time.monotonic() - last_push >= PUSH_INTERVAL:
                    last_push = time.monotonic()
                    yield ui(base + [{"role": "assistant", "content": reply + " ▌"}])
    except Exception as exc:
        reply += f"\n\n**The answer failed.**\n\n{explain_error(exc, model_id)}"
    yield ui(base + [{"role": "assistant", "content": reply.strip() or "_No answer was returned._"}], busy=False)


def handle(message, history, mode, model_id, k, use_web, report):
    text = (message or "").strip()
    if not text:
        gr.Warning("Type a topic or a question first.")
        yield noop()
        return
    base = list(history or []) + [{"role": "user", "content": text}]
    model_id = (model_id or DEFAULT_MODEL).strip()
    if mode == MODE_ASK:
        yield from follow_up(text, base, report or "", model_id)
    else:
        yield from research(text, base, model_id, k, use_web)


def on_stop(history):
    """Close any spinning bubbles and reset the tracker after a cancelled run."""
    fixed = []
    for m in history or []:
        md = m.get("metadata") if isinstance(m, dict) else None
        if md and md.get("status") == "pending":
            m = {**m, "metadata": {**md, "status": "done"}}
        fixed.append(m)
    fixed.append({"role": "assistant", "content": "_Run stopped._"})
    return fixed, RunState().pipeline_html(), gr.update(interactive=True)


def on_new_chat():
    return ui([], pipeline=RunState().pipeline_html(), file=None, busy=False, clear=True, report="")


def on_mode(mode):
    if mode == MODE_ASK:
        return gr.update(placeholder="Ask a question about the last report"), gr.update(value="Ask")
    return gr.update(placeholder="Enter a topic to research"), gr.update(value="Generate report")


# ---------------------------------------------------------------------------
# Look and feel
# ---------------------------------------------------------------------------
THEME = gr.themes.Base(
    primary_hue="blue",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_md,
    font=[gr.themes.GoogleFont("Instrument Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
)

CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,600&display=swap');

:root {
  --rs-desk:#e9edf3; --rs-sheet:#ffffff; --rs-ink:#16202e; --rs-muted:#5e6b7e; --rs-line:#cdd5e1;
  --rs-signal:#2450d6; --rs-signal-ink:#ffffff;
  --rs-live:#d98a00; --rs-live-text:#9a5b00; --rs-ok:#16845b; --rs-err:#c63d3d;
  --rs-serif:"Newsreader", Georgia, "Times New Roman", serif;
}
.dark {
  --rs-desk:#0e141c; --rs-sheet:#161e29; --rs-ink:#e6ebf2; --rs-muted:#93a0b3; --rs-line:#2a3646;
  --rs-signal:#7c9bff; --rs-signal-ink:#0e141c;
  --rs-live:#f2b33d; --rs-live-text:#f2b33d; --rs-ok:#3cc58c; --rs-err:#ff7b7b;
}

body, gradio-app { background: var(--rs-desk) !important; }
footer { display: none !important; }

.gradio-container.gradio-container {
  --body-background-fill: transparent;
  --background-fill-primary: var(--rs-sheet);
  --background-fill-secondary: var(--rs-desk);
  --block-background-fill: transparent;
  --block-border-width: 0px;
  --block-shadow: none;
  --block-label-background-fill: transparent;
  --block-label-text-color: var(--rs-muted);
  --body-text-color: var(--rs-ink);
  --body-text-color-subdued: var(--rs-muted);
  --border-color-primary: var(--rs-line);
  --color-accent: var(--rs-signal);
  --color-accent-soft: color-mix(in srgb, var(--rs-signal) 10%, transparent);
  --link-text-color: var(--rs-signal);
  --chatbot-text-size: 17px;
  --input-background-fill: var(--rs-sheet);
  --input-border-color: var(--rs-line);
  --input-border-color-focus: var(--rs-signal);
  --input-shadow-focus: 0 0 0 3px color-mix(in srgb, var(--rs-signal) 22%, transparent);
  --button-primary-background-fill: var(--rs-signal);
  --button-primary-background-fill-hover: color-mix(in srgb, var(--rs-signal) 86%, #000);
  --button-primary-text-color: var(--rs-signal-ink);
  --button-primary-border-color: transparent;
  --button-secondary-background-fill: transparent;
  --button-secondary-background-fill-hover: color-mix(in srgb, var(--rs-ink) 7%, transparent);
  --button-secondary-text-color: var(--rs-ink);
  --button-secondary-border-color: var(--rs-line);
  --code-background-fill: color-mix(in srgb, var(--rs-ink) 7%, transparent);
  max-width: 1240px !important;
  margin: 0 auto;
  padding-top: 28px !important;
}

/* Header */
.rs-hero { padding: 2px 2px 22px; }
.rs-hero h1 { margin: 0 0 8px; font: 600 clamp(30px, 4vw, 44px)/1.08 var(--rs-serif); letter-spacing: -.02em; color: var(--rs-ink); }
.rs-hero p { margin: 0; max-width: 64ch; font-size: 15.5px; line-height: 1.55; color: var(--rs-muted); }

/* Pipeline rail: a vertical line that fills in as each step finishes */
.rs-rail { position: sticky; top: 16px; align-self: flex-start; }
.rs-ptitle { margin-bottom: 16px; font-size: 14px; font-weight: 600; color: var(--rs-ink); }
.rs-step { position: relative; display: grid; grid-template-columns: 32px 1fr auto; gap: 12px; align-items: start; padding-bottom: 20px; }
.rs-step:last-child { padding-bottom: 0; }
.rs-step::before { content: ""; position: absolute; left: 15px; top: 32px; bottom: 0; width: 2px; background: var(--rs-line); }
.rs-step:last-child::before { display: none; }
.rs-step.done::before { background: var(--rs-ok); }
.rs-node { position: relative; z-index: 1; display: grid; place-items: center; width: 32px; height: 32px; border-radius: 50%;
  border: 2px solid var(--rs-line); background: var(--rs-desk); color: var(--rs-muted); font-size: 13px; font-weight: 600; }
.rs-step.running .rs-node { border-color: var(--rs-live); color: var(--rs-live-text); animation: rs-ring 1.8s ease-out infinite; }
.rs-step.done .rs-node { background: var(--rs-ok); border-color: var(--rs-ok); color: #fff; }
.rs-step.error .rs-node { background: var(--rs-err); border-color: var(--rs-err); color: #fff; }
.rs-meta b { display: block; font-size: 14px; font-weight: 600; line-height: 1.35; color: var(--rs-ink); }
.rs-meta span { font-size: 12.5px; color: var(--rs-muted); }
.rs-step.idle .rs-meta b { color: var(--rs-muted); }
.rs-step.running .rs-meta span { color: var(--rs-live-text); font-weight: 600; }
.rs-step.error .rs-meta span { color: var(--rs-err); }
.rs-time { padding-top: 2px; font-size: 12px; color: var(--rs-muted); font-variant-numeric: tabular-nums; }
@keyframes rs-ring {
  0% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--rs-live) 45%, transparent); }
  100% { box-shadow: 0 0 0 10px transparent; }
}
@media (prefers-reduced-motion: reduce) { .rs-step.running .rs-node { animation: none; } }

/* Conversation: reads like a printed report, set in a serif */
.gradio-container .rs-chat {
  background: var(--rs-sheet); border: 1px solid var(--rs-line); border-radius: 6px;
  box-shadow: 0 1px 2px rgba(22, 32, 46, .06), 0 14px 30px -18px rgba(22, 32, 46, .35);
  font-family: var(--rs-serif); font-size: 17px; line-height: 1.7; color: var(--rs-ink);
}
.gradio-container .rs-chat :is(p, li) { max-width: 72ch; }
.gradio-container .rs-chat :is(h1, h2, h3, h4) { font-family: var(--rs-serif); font-weight: 600; letter-spacing: -.01em; line-height: 1.2; color: var(--rs-ink); }
.gradio-container .rs-chat h1 { font-size: 30px; }
.gradio-container .rs-chat h2 { margin-top: 1.5em; font-size: 23px; }
.gradio-container .rs-chat h3 { font-size: 18px; }
.gradio-container .rs-chat a { color: var(--rs-signal); text-underline-offset: 3px; }
.gradio-container .rs-chat blockquote { border-left: 3px solid var(--rs-line); color: var(--rs-muted); }
.gradio-container .rs-chat :is(button, input, textarea, summary) { font-family: var(--font); }

/* Composer */
#send-btn { font-weight: 600; }
"""

HERO = """
<div class="rs-hero">
  <h1>Research Studio</h1>
  <p>A web search feeds five agents that pick sources, drop the unreliable ones, merge duplicates, analyse the
  findings and write the report. You watch every step stream into the chat, and the models run on your own machine.</p>
</div>
"""

EXAMPLES = [
    ["Latest agentic AI frameworks for Python"],
    ["Open-source LLM releases this month"],
    ["RAG vs fine-tuning: current best practices"],
    ["Latest developments in multimodal AI models"],
]

CHAT_PLACEHOLDER = (
    "## Research anything\n\nEnter a topic below. Each agent's work appears here as it happens, "
    "followed by the finished report. Afterwards, switch to **Ask about the report** to question it."
)

# Gradio 6 moved theme/css from gr.Blocks() to launch() and dropped Chatbot(type=...).
GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6
BLOCKS_KWARGS = {} if GRADIO_6 else {"theme": THEME, "css": CSS}
LAUNCH_KWARGS = {"theme": THEME, "css": CSS} if GRADIO_6 else {}
CHATBOT_KWARGS = {} if GRADIO_6 else {"type": "messages"}

with gr.Blocks(title="Research Studio", **BLOCKS_KWARGS) as demo:
    report_state = gr.State("")
    gr.HTML(HERO)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=270, elem_classes="rs-rail"):
            pipeline = gr.HTML(RunState().pipeline_html())
            with gr.Accordion("Settings", open=False):
                model = gr.Dropdown(
                    choices=MODELS,
                    value=DEFAULT_MODEL,
                    allow_custom_value=True,
                    label="Model",
                    info="Any model you have pulled with Ollama. Larger models follow instructions better.",
                )
                sources = gr.Slider(3, 10, value=6, step=1, label="Sources to fetch")
                use_web = gr.Checkbox(
                    True,
                    label="Search the web",
                    info="Turn off if search is rate-limited. The agents then rely on the model's own knowledge.",
                )
            with gr.Row():
                new_btn = gr.Button("New chat", variant="secondary", size="sm")
                theme_btn = gr.Button("Switch theme", variant="secondary", size="sm")

        with gr.Column(scale=3):
            chat = gr.Chatbot(
                value=[],
                show_label=False,
                height="68vh",
                layout="panel",
                placeholder=CHAT_PLACEHOLDER,
                elem_classes="rs-chat",
                **CHATBOT_KWARGS,
            )
            mode = gr.Radio([MODE_RESEARCH, MODE_ASK], value=MODE_RESEARCH, show_label=False)
            with gr.Row():
                topic = gr.Textbox(
                    show_label=False,
                    placeholder="Enter a topic to research",
                    lines=1,
                    max_lines=4,
                    autofocus=True,
                    scale=6,
                )
                send_btn = gr.Button("Generate report", variant="primary", elem_id="send-btn", scale=2)
                stop_btn = gr.Button("Stop", variant="secondary", scale=1)
            gr.Examples(examples=EXAMPLES, inputs=topic, label="Example topics")
            download = gr.DownloadButton("Download report (Markdown)", visible=False, variant="secondary")

    outputs = [chat, pipeline, download, topic, send_btn, report_state]
    run_event = gr.on(
        triggers=[send_btn.click, topic.submit],
        fn=handle,
        inputs=[topic, chat, mode, model, sources, use_web, report_state],
        outputs=outputs,
    )
    stop_btn.click(on_stop, [chat], [chat, pipeline, send_btn], cancels=[run_event])
    new_btn.click(on_new_chat, None, outputs, cancels=[run_event])
    mode.change(on_mode, mode, [topic, send_btn])
    theme_btn.click(None, None, None, js="() => document.body.classList.toggle('dark')")


if __name__ == "__main__":
    demo.launch(**LAUNCH_KWARGS)
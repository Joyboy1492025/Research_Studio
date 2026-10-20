# Research Studio

A multi-agent research assistant that runs on your own machine. Give it a topic and a web search feeds five AI agents that pick sources, drop unreliable ones, merge duplicates, analyse the findings and write a report. Every step streams live into a chat, and afterwards you can ask follow-up questions about the report.

The language models run locally through [Ollama](https://ollama.com). Only the search query goes to the web.

<!-- Add a screen recording or screenshot here, for example: ![Demo](docs/demo.gif) -->

## Features

- **Five specialised agents** in a sequential [phidata](https://github.com/phidatahq/phidata) workflow
- **Live streaming chat**: each step appears as a collapsible bubble with its own spinner and timer, followed by the finished report
- **Resilient web search**: retries with backoff, spacing between requests and a 10-minute cache, so rate limits don't crash a run
- **Pipeline tracker** showing which step is waiting, working, done or failed
- **Follow-up questions**: switch to "Ask about the report" and the assistant answers only from the report
- **Settings**: choose any Ollama model, set how many sources to fetch, or turn web search off
- **Download** the report as Markdown, **Stop** a run at any time, start a **New chat**, and switch between light and dark themes
- No API keys and no cloud LLM

## How it works

```mermaid
flowchart LR
    T[Topic] --> S[Web search]
    S --> C[Research Collector]
    C --> V[Source Validator]
    V --> D[Deduplication Expert]
    D --> A[Research Analyst]
    A --> J[Senior Journalist]
    J --> R[Report]
    R -.-> F[Follow-up Q&A]
```

| Step | What it does |
| --- | --- |
| Web search | Fetches news and web results with `ddgs`, retrying when the search engine refuses |
| Research Collector | Picks the relevant sources and returns title, summary, date and URL |
| Source Validator | Checks credibility and removes unreliable sources |
| Deduplication Expert | Merges repeated stories |
| Research Analyst | Identifies trends, compares sources and extracts insights |
| Senior Journalist | Writes the report with an executive summary, sections and references |

Design choices worth knowing:

- **Search is plain Python, not a model tool call.** Small local models are unreliable at tool calling, and agents without tools stream their replies token by token on every Ollama version.
- **Progress travels through `RunResponse.event`.** The workflow yields events such as `stage_start:2` and `stage_chunk:2`, and the UI turns them into chat bubbles and tracker updates.
- **The agents are defined once** in the `STAGES` list, which drives both the workflow and the pipeline tracker.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/download) installed and running
- An internet connection for web search

## Installation

```bash
git clone https://github.com/<your-username>/research-studio.git
cd research-studio

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -U ddgs phidata ollama gradio

ollama pull llama3.2:3b
```

## Usage

```bash
python research_studio.py
```

Open the address printed in the terminal (usually http://127.0.0.1:7860).

1. Enter a topic and select **Generate report**.
2. Watch each step stream into the chat. The report is the last message.
3. Select **Download report (Markdown)** to save it.
4. Switch the mode to **Ask about the report** to question it.

Reports are saved to your system temp folder as `report-<date>-<topic>.md`.

### Settings

| Setting | Default | Notes |
| --- | --- | --- |
| Model | `llama3.2:3b` | Any model you have pulled with Ollama. Larger models such as `llama3.1:8b` or `qwen2.5:7b` follow instructions better |
| Sources to fetch | 6 | Between 3 and 10 |
| Search the web | On | Turn off if search is rate-limited. The agents then rely on the model's own knowledge and say so |

### Configuration

Environment variable:

| Variable | Purpose |
| --- | --- |
| `SEARCH_PROXY` | Proxy URL for search requests, for example `http://user:pass@host:port` |

Constants at the top of `research_studio.py`:

| Constant | Purpose |
| --- | --- |
| `DEFAULT_MODEL`, `MODELS` | Default model and the dropdown choices |
| `RETRY_DELAYS` | Seconds to wait after each failed search attempt |
| `MIN_GAP` | Minimum seconds between two search requests |
| `CACHE_TTL` | How long search results are reused for the same topic |
| `PUSH_INTERVAL` | How often the UI refreshes while text streams |

To change an agent's behaviour, edit its `instructions` in the `STAGES` list.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| "Web search is rate-limited" or `RatelimitException` | Wait a minute and retry, turn off **Search the web**, set `SEARCH_PROXY`, and make sure you use the maintained package: `pip install -U ddgs` |
| Connection refused or model not found | Start Ollama with `ollama serve` and pull the model with `ollama pull <model>` |
| Step bubbles look flat or the chat layout is off | Upgrade Gradio: `pip install -U gradio` |
| Poor or thin results | Switch to a larger model in Settings |

## Limitations

- Language models can make mistakes. Check important claims against the linked sources.
- `ddgs` queries public search engines without a service guarantee, so availability and result quality can vary.
- The search cache lives in memory and resets when the app restarts.
- Runs share one Ollama instance, so concurrent runs are slow on modest hardware.

## Tech stack

Python, [phidata](https://github.com/phidatahq/phidata), [Ollama](https://ollama.com), [ddgs](https://pypi.org/project/ddgs/), [Gradio](https://www.gradio.app).

## License

Choose a license and add a `LICENSE` file.
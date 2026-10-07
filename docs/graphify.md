# Graphify for JBDtection

The project uses the official `graphifyy` package (CLI: `graphify`), upgraded
from 0.9.69 to 0.9.70 on 2026-09-28. The persistent graph includes symbols,
functions, call/import/inheritance relationships, comments and supported local
configuration/document extractors. It is more than a class-name index.

## Small context, full persistent index

Start with a focused operation. This wrapper defaults to a 1,200-token query
budget and enforces a separate 4,800-character output limit, including its
truncation notice. Character limits are not exact tokenizer counts.

```bash
python3 scripts/graphify_context.py explain 'UnifiedPdfProcessor'
python3 scripts/graphify_context.py path 'PatternMatcher' 'UnifiedPdfProcessor'
python3 scripts/graphify_context.py affected 'PatternMatcher'
python3 scripts/graphify_context.py query 'UnifiedPdfProcessor process_pdf'
```

Use symbols from the graph in queries, even when the conversation is in Persian.
If a result is truncated, narrow the question or pass the printed `--offset`.
Do not treat a partial result as an exhaustive dependency list. `explain` itself
summarizes neighbors; MCP `get_neighbors` supports relation filters.

The existing Codex MCP configuration already enables `graphify-mcp`. Tools
include `get_node`, `get_neighbors`, `query_graph`, `shortest_path`, and
`graph_stats`. Prefer `token_budget=1200`, depth 1–2 and relation filters.
The server reloads changed graph files on tool calls. A new Codex session picks
up the upgraded server executable; the CLI wrapper works in this session too.

## Automatic updates and server limits

Git post-commit/post-checkout hooks rebuild the graph locally. A user systemd
service also watches uncommitted saves, with a 10-second debounce, one extraction
worker, lower CPU priority, a one-core CPU quota and a 4 GiB memory ceiling.
Manual refresh defaults to two workers. Server resources detected: two CPUs,
31 GiB RAM and a Tesla K80 with approximately 11 GiB VRAM.

```bash
systemctl --user status graphify-watch
journalctl --user -u graphify-watch -n 30 --no-pager
systemctl --user restart graphify-watch
bash scripts/graphify_refresh.sh
```

The service starts with the user's systemd manager. Without user lingering,
starting at machine boot before login is not guaranteed. Code edits update the
graph/report automatically; run the refresh script to regenerate the additional
wiki/tree/call-flow exports. Unsupported document changes can flag a pending
semantic update instead of invoking a model automatically.

## Browse it yourself

Open these generated files in a browser or Markdown viewer:

- `graphify-out/graph.html`: searchable graph with node relationships.
- `graphify-out/GRAPH_TREE.html`: collapsible files and symbols.
- `graphify-out/JBDtection-callflow.html`: architecture/call-flow diagrams.
- `graphify-out/wiki/index.md`: generated navigation and component articles.

## Semantic deep mode

Local AST indexing and graph traversal do not consume LLM API tokens.
`--mode deep` affects the semantic extraction pass; it is not a switch that
adds implementation details to every code node. Semantic extraction of
documents/media needs a configured backend. No backend credential or running
local model was detected, so no paid semantic calls were performed.

Once a backend has been configured, run a deliberate semantic build, keeping
the existing graph backed up:

```bash
graphify extract . --mode deep --max-workers 2 --max-concurrency 1 --token-budget 4000
```

Backend/model suitability and cost depend on what is configured. Avoid `--force`
for routine semantic updates: the incremental manifest and caches reduce repeat
work. Keep customer input/output files excluded. Inferred graph edges do not
prove exact runtime paths; inspect the smallest source range when behavior matters.

The built-in benchmark compares graph retrieval with reading the whole corpus.
It is an estimate, not a measurement of your billed conversation tokens.

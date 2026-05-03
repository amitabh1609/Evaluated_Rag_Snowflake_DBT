# What I Got Wrong — Running Log

> This is a scratch file. At the end of Phase 5 it gets curated into the README's
> "What didn't work" section. Entries go in as soon as something breaks or a direction
> changes — do not batch this.

## Format

```
### [Phase N — YYYY-MM-DD] Title
**What I tried:** ...
**What happened:** ...
**Root cause:** ...
**What I did instead:** ...
**Would I do it differently?** ...
```

---

### [Phase 3 — 2026-05-03] Citation parser reconstructing full chunk_id
**What I tried:** Regex `[doc_id:chunk_suffix]` captured the two groups and looked up `chunk_suffix` directly in the payload dict.
**What happened:** Lookup always missed — the payload dict is keyed by the full `"doc_id:chunk_suffix"` string (how chunk_ids are stored), not by the suffix alone.
**Root cause:** The LLM writes `[sf_wh:3]` as a citation; the regex captures `doc_id=sf_wh` and `chunk_suffix=3` separately, but Qdrant/BM25 stored the chunk_id as `sf_wh:3`. Also had a second bug: the chunk_lookup builder was adding a double-encoded key `"sf_wh:sf_wh:3"`.
**What I did instead:** Reconstruct `full_chunk_id = f"{doc_id}:{chunk_suffix}"` before lookup, and removed the double-encoded key from the builder.
**Would I do it differently?** Yes — I'd write the citation parser unit test before the integration rather than after.

### [Phase 2 — 2026-05-03] tiktoken model_name vs encoding_name in RecursiveCharacterTextSplitter
**What I tried:** `from_tiktoken_encoder(model_name="cl100k_base", …)` — passing the encoding name as a model name.
**What happened:** `KeyError: 'Could not automatically map cl100k_base to a tokeniser'` — tiktoken's model→encoding map does not contain encoding names.
**Root cause:** `from_tiktoken_encoder` calls `tiktoken.encoding_for_model()` which expects a model name like `gpt-4` or `text-embedding-ada-002`, not an encoding name.
**What I did instead:** Switched to `encoding_name="cl100k_base"` which calls `tiktoken.get_encoding()` directly.
**Would I do it differently?** Read the langchain_text_splitters source before assuming the parameter name maps 1:1 to the tiktoken API.

# The Story of `ingest_v2.py` — How Documents Become Searchable Knowledge

Imagine you have a pile of earnings call transcripts and SEC filings sitting in a folder. Raw text files, dense PDFs. A language model can't search through them efficiently on its own — it needs those documents pre-processed, sliced into pieces, and stored in a way it can retrieve later. That is the entire job of `ingest_v2.py`. It runs once (or whenever you add new files), and by the time it's done, every document is living in Pinecone's cloud as a searchable vector. Here's how that journey unfolds, function by function.

---

## Act 1 — The Opening Scene: `main()`

Everything starts in `main()`. Think of it as the stage director — it doesn't do the hard work itself, but it calls each actor onto the stage in the right order.

The director's script looks like this:

1. **Connect** to Pinecone (the cloud database).
2. **Load** all the documents from the `data/` folder.
3. **Chunk** those documents into digestible pieces.
4. **Embed and store** those pieces in the cloud.

If no documents are found at step 2, the director calls it a night early — there's nothing to process.

---

## Act 2 — Knocking on Pinecone's Door: `get_pinecone_index()`

Before any documents can be stored, the pipeline needs a place to put them. `get_pinecone_index()` is the character that walks up to Pinecone's cloud and says: *"Is our index already there, or do we need to build it?"*

It first grabs the `PINECONE_API_KEY` from the environment — without this, the door doesn't open. Then it lists all existing indexes in the account. If the index named `earningslens-v2` is already there, it simply returns a handle to it. If not, it creates a brand-new one with these specifications:

- **384 dimensions** — because the embedding model (`all-MiniLM-L6-v2`) outputs 384-dimensional vectors.
- **Cosine similarity** — the distance metric used when comparing vectors during search.
- **Serverless on AWS `us-east-1`** — the free-tier hosting option.

This function ensures the index exists no matter what, so the rest of the pipeline never has to worry about it.

---

## Act 3 — Naming Every Piece: `make_chunk_id(source, chunk_index)`

Every chunk that gets stored in Pinecone needs a unique name tag. This is the function that stamps those tags.

It takes the source filename (e.g., `apple_q3_2024.txt`) and a chunk index number, then produces a stable ID like:

```
a3f9c2b1d4e8-chunk-0042
```

The first part is the first 12 characters of the SHA-256 hash of the filename. The second part is the chunk's sequential number, zero-padded to 4 digits.

Why bother with this? Because Pinecone upserts are **idempotent by ID** — if you run `ingest_v2.py` twice on the same files, the same chunk IDs get written again, simply overwriting what was already there. No duplicates pile up in the index.

---

## Act 4 — Reading the Transcripts: `load_txt_documents(data_dir)`

The first type of document the pipeline handles is plain `.txt` files — earnings call transcripts. This function scans the `data/` directory (and all subdirectories) for every `.txt` file it can find.

For each file it finds, it uses LangChain's `TextLoader` to read the raw text. Then it stamps each document with two metadata fields:

- `source` — the filename (e.g., `apple_q3_2024.txt`)
- `doc_type` — set to `"transcript"` so the rest of the system knows what kind of document it's dealing with.

All loaded transcripts are collected into a single list and returned.

---

## Act 5 — Two Helper Characters for PDFs

Before the main PDF loader enters, two small helper functions do quiet but important work behind the scenes.

### `_format_table(table)`

PDFs often contain financial tables — revenue breakdowns, balance sheets, earnings summaries. When `pdfplumber` extracts a table, it comes back as a list of rows, each row a list of cells. This helper flattens that structure into a human-readable string where cells are separated by ` | ` and rows by newlines.

For example:
```
Revenue | 89.5B | 90.1B
Net Income | 21.7B | 22.3B
```

This formatted text gets appended to the page's content so that the model can reason about financial figures that live inside tables.

### `_is_boilerplate(text)`

SEC filings are notoriously padded with boilerplate — EDGAR headers, cover pages, tables of contents, repeated regulatory disclaimers. Feeding those into the vector store would pollute search results with useless noise.

This function scans a page's text for a list of known boilerplate phrases (like `"table of contents"` or `"united states securities and exchange commission"`). If **2 or more** of those phrases appear, the page is flagged as boilerplate and skipped entirely.

---

## Act 6 — Reading the SEC Filings: `load_pdf_documents(data_dir)`

This is the most complex loading function because PDFs are messy. It finds every `.pdf` in the `data/` folder and processes each one page by page using `pdfplumber`.

For each page, the process is:

1. **Extract raw text** from the page.
2. **Extract tables** — if any exist, format them with `_format_table()` and append them to the page text under a `[TABLE]` marker.
3. **Filter out junk** — if the combined text is under 100 characters (practically empty) or triggers `_is_boilerplate()`, skip the page.
4. **Create a Document** — if the page survives those filters, wrap it into a LangChain `Document` object with metadata: `source` (filename), `page_number`, and `doc_type` set to `"pdf"`.

At the end, it reports how many pages were extracted versus skipped for each file.

---

## Act 7 — Slicing Everything Up: `chunk_documents(documents)`

At this point, all the transcripts and PDF pages are loaded. But individual documents can be very long — a full earnings call transcript could run tens of thousands of characters. Language models and vector searches work best on focused, bite-sized pieces.

`chunk_documents()` feeds all documents through LangChain's `RecursiveCharacterTextSplitter`, which slices them up using these settings:

- **Chunk size**: 1,000 characters — a chunk won't exceed this.
- **Overlap**: 200 characters — consecutive chunks share 200 characters at their boundary, so context isn't lost at a split point.
- **Separator priority**: The splitter tries to break at `\n\n` (paragraph), then `\n` (newline), then `. ` (sentence), then space, then hard character split as a last resort.

The result is a flat list of many small chunks, each carrying the metadata inherited from its parent document.

---

## Act 8 — The Grand Finale: `create_vector_store(chunks, index)`

This is the climax — where all those chunks get transformed from plain text into mathematical vectors and uploaded to Pinecone.

**Step 1 — Load the embedding model.** The `all-MiniLM-L6-v2` model from HuggingFace is loaded onto the CPU. This model converts any text string into a 384-dimensional vector — a point in high-dimensional space that captures the semantic meaning of the text.

**Step 2 — Assign IDs.** Every chunk gets its stable ID from `make_chunk_id()`, building the list of IDs that will accompany each vector in Pinecone.

**Step 3 — Upsert.** LangChain's `PineconeVectorStore.from_documents()` takes all the chunks, embeds them using the model, and sends them to Pinecone in batches. Each vector is stored alongside the chunk's text and metadata so the app can retrieve and display the source later.

**Step 4 — Verify.** After upserting, the function calls `index.describe_index_stats()` to confirm how many total vectors now live in the index and prints that count. This is the final confirmation that the pipeline succeeded.

---

## The Epilogue

When `main()` finishes, every document in `data/` has been:

1. Loaded and cleaned
2. Sliced into ~1,000-character chunks
3. Converted into 384-dimensional semantic vectors
4. Stored permanently in Pinecone under a stable, deduplicated ID

From this moment on, `app_v2.py` can search those vectors at query time — finding the most semantically relevant chunks to any question a user asks, and feeding them into the language model to generate grounded, accurate answers.

The pipeline can be re-run safely at any time. New documents get added; existing ones get silently overwritten. Nothing duplicates.

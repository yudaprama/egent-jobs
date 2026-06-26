# egent-jobs

River-backed job worker that replaces the Next.js BFF polling loops in
`lobehub/apps/server/src/routers/async/{file,image,video}.ts`. River is
Postgres-native — no Redis for jobs — so it reuses the existing Supabase pool.

## Shipped workers

| Kind | Replaces | Status |
|---|---|---|
| `embed_file_chunks` | `async/file.ts:embeddingChunks` | Shipped |
| `parse_file_to_chunks` | `async/file.ts:parseFileToChunks` | Shipped (default `TextChunker`; swap in unstructured.io bridge for PDF/DOCX) |
| `create_image` | `async/image.ts:createImage` | Roadmap |
| `create_video` | `async/video.ts:createVideo` | Roadmap |

## BFF producer

`apps/server/src/server/rivers/riverProducer.ts` exposes `enqueueEmbedFileChunks` / `enqueueParseFileToChunks`. The producer is **lazy-initialised** (first call pulls the cached drizzle instance via `getServerDB()`) and guarded by `isRiverHealthy()` so the BFF automatically falls back to the legacy `createAsyncCaller()` self-HTTP path when `egent-jobs` hasn't been deployed yet or River migrations haven't run.

`ChunkService.asyncEmbeddingFileChunks` and `asyncParseFileToChunks` (in `apps/server/src/services/chunk/index.ts`) are now the producer-side entry points — they create the `async_tasks` row, stamp the foreign key on `public.files`, and INSERT a River job. No HTTP self-call when River is healthy.

## Run

```bash
export LOBEHUB_PG_DSN='postgres://...@host:5432/lobehub'
export OPENAI_API_KEY=sk-...
# optional:
#   OPENAI_EMBEDDINGS_MODEL=text-embedding-3-small
#   OPENAI_EMBEDDINGS_URL=https://api.openai.com/v1
#   EGENT_JOBS_PORT=10540
#   EMBEDDING_BATCH_SIZE=10
#   EMBEDDING_CONCURRENCY=3
#   EGENT_JOBS_MAX_WORKERS=10
#   EGENT_JOBS_MIGRATE=1
make            # builds ./bin/egent-jobs
./bin/egent-jobs
```

Health: `GET /healthz` → 200 `ok`. Version: `GET /version`.

## Architecture

```
producer (BFF / CLI)  ──►  river_job (Postgres)  ──►  egent-jobs worker
                                   │                          │
                                   │                          ├─ MarkProcessing (async_tasks)
                                   │                          ├─ fetch chunks (chunks + file_chunks)
                                   │                          ├─ EmbedBatch (OpenAI-compat HTTP)
                                   │                          ├─ bulk INSERT (public.embeddings, ON CONFLICT UPDATE)
                                   │                          └─ MarkSuccess / MarkError (async_tasks)
                                   │
                            async_tasks stays as the
                            user-visible status ledger
```

`async_tasks` remains the status ledger the BFF reads via Tier 1 pREST CRUD.
River's own rows live in `river_job` (separate schema, auto-migrated).

## Packages

- `asynctask/` — status enum + thin `Store` mirroring `asyncTaskModel.update()`
- `embeddings/` — `Embedder` interface + OpenAI-compatible HTTP impl
- `fileingest/` — the `EmbedFileChunksWorker` (River `JobArgs` + `Work`)
- `queue/` — River client wrapper, DSN sanitizer, pgvector probe
- `main.go` — CLI entry point with graceful shutdown

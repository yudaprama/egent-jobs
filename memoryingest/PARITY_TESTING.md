# Memory Extraction Parity Testing

This guide covers testing the Go extraction against the TS version to validate parity before retiring the TS router.

## Phase 1: Single-Topic Trace (Real OpenAI)

**What it does:** Extracts a single realistic chat topic using real OpenAI API, captures full LLM call traces, and outputs JSON for manual comparison with TS.

**Prerequisites:**
- `OPENAI_API_KEY` environment variable (OpenAI account with API access)
- ~$0.50 budget for a few test extractions

**Run:**
```bash
cd egent-jobs
OPENAI_API_KEY=sk-... go test -v -run TestPhase1SingleTopicTrace ./memoryingest
```

**Output:**
- `phase1_extraction_trace.json` — full extraction trace with:
  - Input content + topic ID
  - All LLM call traces (layer, request, response, duration)
  - Extracted memories grouped by layer
  - Counts per layer

**Next steps:**
1. Review the JSON output:
   ```bash
   cat phase1_extraction_trace.json | jq '.extracted_memories'
   ```
2. Manually compare counts and memory content to the TS version
   - Same gatekeeper decision (relevant/not)?
   - Similar extraction counts per layer (±2 is okay due to LLM variance)?
   - Semantic similarity of field values?
3. If satisfied, move to Phase 2

## Phase 2: Batch Extraction (Mocked LLM)

**What it does:** Runs batch extraction on multiple topics with mocked LLM responses. Validates worker robustness without API costs.

**Prerequisites:**
- None (uses canned LLM responses)

**Run:**
```bash
cd egent-jobs
go test -v -run TestPhase1MockBatch ./memoryingest
```

**Output:**
- Pass/fail on 3 synthetic topics (personal, small talk, work project)
- Validates:
  - Gatekeeper approval logic
  - Multi-topic batch handling
  - Error handling

**Expected behavior:**
- `personal_experience` topic: extracts from all 5 layers
- `small_talk` topic: gatekeeper rejects, no extraction
- `work_project` topic: extracts from all 5 layers

## Phase 3: Database Integration (Test DB Required)

**Coming soon** — will validate full persistence to a test database, including:
- Parent/child row creation
- Vector embeddings (pgvector compatibility)
- Search functionality

## Debugging

### Inspect LLM calls
```bash
cat phase1_extraction_trace.json | jq '.llm_calls[] | {layer, duration_ms, response}'
```

### Check extraction counts
```bash
cat phase1_extraction_trace.json | jq '{
  identities: (.extracted_memories.identities | length),
  activities: (.extracted_memories.activities | length),
  contexts: (.extracted_memories.contexts | length),
  experiences: (.extracted_memories.experiences | length),
  preferences: (.extracted_memories.preferences | length)
}'
```

### View extracted memory details
```bash
cat phase1_extraction_trace.json | jq '.extracted_memories.identities[] | {description, type, role}'
```

## Troubleshooting

### "OPENAI_API_KEY not set"
Set the environment variable:
```bash
export OPENAI_API_KEY=sk-your-key-here
```

### "status 401: Invalid authentication"
Check that your OpenAI API key is valid and has active credits.

### "No memories extracted (gatekeeper rejected)"
This is normal if the test content isn't deemed relevant by the LLM gatekeeper. Try different input content with more personal/durable information.

### LLM timeout
The test uses a 30-second timeout per LLM call. If OpenAI is slow:
1. Retry after a few minutes
2. Check OpenAI status page (status.openai.com)

## Comparing to TS

To compare Go extraction with TS:

1. **If TS extractor is still running:**
   - Use the same chat topic as Phase 1 input
   - Run TS extraction via the LobeHub backend
   - Compare `phase1_extraction_trace.json` counts and field values

2. **If TS extractor is offline:**
   - Check async task records in the DB (table: `async_tasks`)
   - Extract the error payload or result to see what TS extracted
   - Compare counts and memory fields

3. **Metrics to compare:**
   - Gatekeeper decision (relevant: true/false) — should match 100%
   - Extraction counts per layer — should match ±2 (LLM variance)
   - Field values — should be semantically similar (not word-for-word)
   - NULL embedding rate — should be <5% (only for blank fields)

## Known Differences

### Single vs. Multi-vector embedding
- TS generates multiple embeddings per memory (summary + details + field-specific)
- Go v1 generates one embedding (primary field only)
- **Impact:** Marginal. Multi-vector search is nice but single-vector works for MVP.

### Memory schema subset
- Go stores minimal fields (`description`, `narrative`, `situation`, etc.)
- TS stores richer schema (`memory_type`, `memory_category`, `metadata` with message IDs)
- **Impact:** Go memoryingest focuses on core extraction; palace DB layer owns the rest.

### No observability
- Go has no S3 trace recording (logs only)
- TS has full observability (S3 traces + OTLP metrics)
- **Impact:** Acceptable for River workers; logs are enough for debugging.

## Next: Retire TS Router

Once Phase 1, 2, and 3 pass with <5% variance:

1. Remove TS `MemoryExtractionExecutor` and related classes
2. Remove `MemoryExtractionService` dependency from lobehub
3. Update frontend to remove `userMemories.searchMemory` router calls (already migrated to palace search)
4. Verify CI passes
5. Deploy

Timeline: 4 weeks (1 week per phase, 1 week buffer).

# Evaluation Workflow

This workflow evaluates whether the knowledge base Q&A bot retrieves the right source and returns grounded citations for the questions in `eval/evaluation_set.json`.

## Goals

Measure three retrieval and grounding metrics:

- `recall@k`: whether the expected source appears anywhere in the top `k` retrieved sections or chunks.
- `MRR`: how highly the expected source appears in the ranked retrieval results.
- `source_hit_rate`: whether the final `/chat` response cites the expected source.

Evaluate positive cases separately from negative cases. Positive cases have an `expected_source`. Negative cases have `expect_fallback: true` and should not retrieve or cite a forced source.

## Inputs

- Knowledge base docs: `docs/*.md`
- Evaluation set: `eval/evaluation_set.json`
- Candidate implementation:
  - Markdown KB: `markdown_kb`
  - Vector RAG: `vector_rag`

Each positive eval case must include:

```json
{
  "id": "refund_timeline_core",
  "query": "How long do refunds take?",
  "expected_source": "refund_policy.md#refund-timeline",
  "expect_fallback": false
}
```

Each negative eval case must include:

```json
{
  "id": "negative_shipping_cost",
  "query": "How much does standard shipping cost?",
  "expected_source": null,
  "expect_fallback": true
}
```

## Source ID Normalization

Use the same source ID format everywhere:

```text
filename.md#heading-slug
```

Examples:

- `refund_policy.md#refund-timeline`
- `shipping_faq.md#tracking-number`
- `account_help.md#reset-password`

For Markdown KB retrieval, the section object already has `section.id`.

For Vector RAG retrieval, combine metadata:

```python
source_id = f"{doc.metadata['source']}#{doc.metadata['heading']}"
```

## Metrics

### Recall@k

For each positive case, inspect the top `k` retrieved source IDs.

```text
hit@k = 1 if expected_source is in retrieved_sources[:k], else 0
recall@k = sum(hit@k) / number_of_positive_cases
```

Recommended values:

- `recall@1`
- `recall@3`
- `recall@5`

Use `recall@3` as the main retrieval score because `/chat` currently retrieves `k=3`.

### MRR

MRR rewards the expected source appearing earlier in the ranked list.

```text
rank = 1-based position of expected_source in retrieved_sources
reciprocal_rank = 1 / rank if found, else 0
MRR = average(reciprocal_rank across positive cases)
```

Example:

```text
retrieved_sources = [
  "shipping_faq.md#standard-shipping",
  "refund_policy.md#refund-timeline",
  "account_help.md#reset-password"
]
expected_source = "refund_policy.md#refund-timeline"

rank = 2
reciprocal_rank = 0.5
```

### Source Hit Rate

Source hit rate evaluates the full `/chat` response, not only retrieval.

For each positive case:

```text
source_hit = 1 if expected_source appears in response.sources, else 0
source_hit_rate = sum(source_hit) / number_of_positive_cases
```

This catches failures where retrieval found the right source but the final response omitted or changed the citation.

## Negative Case Checks

Negative cases should measure fallback behavior, not recall.

For each negative case:

```text
fallback_hit = 1 if response.answer says it cannot confirm from the knowledge base, else 0
no_source_hit = 1 if response.sources is empty, else 0
```

Report:

```text
fallback_rate = sum(fallback_hit) / number_of_negative_cases
negative_no_source_rate = sum(no_source_hit) / number_of_negative_cases
```

These metrics are especially useful after adding a retrieval score threshold.

## Workflow

### 1. Build or Load the Index

Start from a clean server state when comparing changes.

For Markdown KB:

```bash
cd markdown_kb
export OPENAI_API_KEY="sk-..."
uvicorn app.main:app --reload
curl -X POST http://localhost:8000/index
```

For Vector RAG:

```bash
cd vector_rag
export OPENAI_API_KEY="sk-..."
uvicorn app.main:app --reload
curl -X POST http://localhost:8000/index
```

### 2. Run Direct Retrieval Evaluation

Direct retrieval evaluation should call `indexer.search(query, k=5)` without invoking the LLM.

For each positive eval case:

1. Read `query` and `expected_source`.
2. Run retrieval with `k=5`.
3. Convert each retrieved item into a normalized source ID.
4. Record the ordered source IDs and scores.
5. Compute `recall@1`, `recall@3`, `recall@5`, and reciprocal rank.

Recommended per-case output:

```json
{
  "id": "refund_timeline_core",
  "query": "How long do refunds take?",
  "expected_source": "refund_policy.md#refund-timeline",
  "retrieved_sources": [
    "refund_policy.md#refund-timeline",
    "refund_policy.md#cancellation-window",
    "shipping_faq.md#standard-shipping"
  ],
  "scores": [2.41, 0.72, 0.31],
  "hit_at_1": true,
  "hit_at_3": true,
  "hit_at_5": true,
  "reciprocal_rank": 1.0
}
```

### 3. Run API Source Evaluation

API source evaluation should call `/chat` for every eval case.

For each positive eval case:

1. Send the query to `/chat`.
2. Collect `answer` and `sources`.
3. Normalize returned sources to `filename.md#heading-slug`.
4. Check whether `expected_source` appears in returned sources.
5. Check that the answer does not fallback.

For each negative eval case:

1. Send the query to `/chat`.
2. Check that the answer uses the fallback behavior.
3. Check that `sources` is empty.

Recommended per-case output:

```json
{
  "id": "shipping_tracking_paraphrase_1",
  "query": "How are customers notified of shipment tracking?",
  "expected_source": "shipping_faq.md#tracking-number",
  "returned_sources": [
    "shipping_faq.md#tracking-number"
  ],
  "source_hit": true,
  "fallback": false
}
```

## Final Report

Produce one summary per implementation and configuration.

Example:

```json
{
  "implementation": "markdown_kb",
  "retrieval_k": 5,
  "chat_k": 3,
  "positive_cases": 27,
  "negative_cases": 10,
  "metrics": {
    "recall@1": 0.89,
    "recall@3": 0.96,
    "recall@5": 1.0,
    "mrr": 0.93,
    "source_hit_rate": 0.96,
    "fallback_rate": 0.9,
    "negative_no_source_rate": 0.9
  }
}
```

Also save failed cases so they can be inspected:

```json
{
  "failed_retrieval_cases": [
    "shipping_standard_paraphrase_2"
  ],
  "failed_source_hit_cases": [
    "refund_non_refundable_paraphrase_1"
  ],
  "failed_negative_cases": [
    "negative_return_window"
  ]
}
```

## Pass Criteria

Use these as starting targets for the small sample KB:

- `recall@3 >= 0.95`
- `MRR >= 0.90`
- `source_hit_rate >= 0.95`
- `fallback_rate >= 0.90`
- `negative_no_source_rate >= 0.90`

If `recall@3` is high but `source_hit_rate` is low, the retrieval layer is probably working and the response/citation formatting needs attention.

If `recall@3` is low, inspect tokenization, heading matching, chunking, embeddings, or score thresholds.

If negative cases return sources too often, tighten the retrieval score threshold or improve fallback instructions.

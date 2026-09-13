# Zero-shot GLiNER2 span probe (no ASTE training)

**Provenance.** Output of `zeroshot_span_probe` in
`archive/2026-09-10-explorations/research/modal_research_full.py`, run on Modal on
2026-09-09 (~10:50 UTC). The function returned the numbers but never wrote them to
disk; this file was recovered verbatim from that session's transcript on 2026-09-13.

Setup: `fastino/gliner2-base-v1`, `batch_extract_entities` with GLiNER2's own schema
interface, threshold 0.3, first 200 test sentences per dataset, exact span match.
Nothing is trained. GLiNER2 used this way returns spans only - it has no ASTE
triplet output, so no zero-shot triplet F1 exists.

```
  14lap  aspect_term/opinion_term aspect F1=0.582 (P=0.554 R=0.614)   opinion F1=0.442
  14lap  aspect/opinion           aspect F1=0.585 (P=0.566 R=0.606)   opinion F1=0.477
  14lap  feature/sentiment_expr   aspect F1=0.577 (P=0.665 R=0.509)   opinion F1=0.398
  14lap  target/opinion_word      aspect F1=0.512 (P=0.470 R=0.563)   opinion F1=0.551
  14res  aspect_term/opinion_term aspect F1=0.698 (P=0.681 R=0.715)   opinion F1=0.666
  14res  aspect/opinion           aspect F1=0.723 (P=0.718 R=0.727)   opinion F1=0.627
  14res  feature/sentiment_expr   aspect F1=0.616 (P=0.876 R=0.475)   opinion F1=0.553
  14res  target/opinion_word      aspect F1=0.757 (P=0.728 R=0.788)   opinion F1=0.707
  15res  aspect_term/opinion_term aspect F1=0.488 (P=0.460 R=0.521)   opinion F1=0.532
  15res  aspect/opinion           aspect F1=0.547 (P=0.505 R=0.596)   opinion F1=0.595
  15res  feature/sentiment_expr   aspect F1=0.505 (P=0.763 R=0.377)   opinion F1=0.524
  15res  target/opinion_word      aspect F1=0.618 (P=0.548 R=0.709)   opinion F1=0.670
  16res  aspect_term/opinion_term aspect F1=0.553 (P=0.534 R=0.572)   opinion F1=0.547
  16res  aspect/opinion           aspect F1=0.620 (P=0.600 R=0.642)   opinion F1=0.598
  16res  feature/sentiment_expr   aspect F1=0.451 (P=0.681 R=0.337)   opinion F1=0.518
  16res  target/opinion_word      aspect F1=0.657 (P=0.577 R=0.761)   opinion F1=0.687

  avg over domains, by phrasing (mean of aspect and opinion F1):
    target/opinion_word      0.645
    aspect/opinion           0.597
    aspect_term/opinion_term 0.563
    feature/sentiment_expr   0.518
```

Best phrasing (`target` / `opinion word`), averaged over the four datasets:
aspect F1 **0.636**, opinion F1 **0.654**.

Supervised reference on the same first-200 test sentences (spans read off predicted
triplets, recomputed 2026-09-13):

| model | aspect F1 | opinion F1 |
|---|---:|---:|
| zero-shot GLiNER2, best phrasing | 0.636 | 0.654 |
| A0, generic span-pair head, seeds 42-46 | 0.879 | 0.857 |
| GLiNER2-ASTE (`B2h-cf`), seeds 62-66 | 0.897 | 0.873 |
| RelAdapt-GLiNER2 (`R1bm-cf`), seeds 62-66 | 0.896 | 0.874 |

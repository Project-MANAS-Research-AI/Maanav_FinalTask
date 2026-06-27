# BPE Implementation — Part B Analysis
## BPE-Dropout as a Novel Addition to Vanilla BPE

---

## 1. What is Wrong with Vanilla BPE?

Vanilla BPE always produces the same tokenization for the same input. Every word gets split the exact same way every single time. For common words this is fine since the model sees them enough to learn good representations. But for rare or unseen words, that one fixed split might just be wrong or unhelpful, and the model never gets a chance to see any other possibility.

This also means the model never learns that subword boundaries are somewhat flexible. It never sees that "running" and "run" share a stem through their tokenization, because the splits never vary. The model just memorizes the specific pieces it was given.

BPE-Dropout (Provilkov et al., 2020) and Subword Regularization (Kudo, 2018) both target exactly this problem by making the segmentation stochastic during training.

---

## 2. What We Change

The vocabulary and merge table stay exactly the same as vanilla BPE. The only change is during encoding — each merge rule gets randomly skipped with some probability p. So the same word might tokenize differently on different calls, giving the model exposure to multiple plausible splits without changing anything about training the tokenizer itself.

With p = 0.0 the behaviour is completely identical to vanilla BPE. With p = 0.1 roughly 10% of merges get skipped, producing slightly more fragmented but more varied outputs. At inference time you just set p back to 0 and get normal deterministic encoding.

---

## 3. What Experiment Would Prove It Works?

Both models use the same corpus with a 90/10 train/test split. The test set is never touched during training. The only difference between the two runs is the dropout probability.

The metrics we care about are whether decode(encode(text)) still returns the original string, compression ratio (characters per token), fertility (tokens per word), UNK rate, fragmentation on rare words, and wall-clock time for both training and encoding.

The real proof would actually be training a small language model on top and comparing perplexity. Tokenizer-level metrics can only tell us so much — the dropout benefit shows up downstream, not in compression numbers.

---

## 4. Results

```
Metric                        Basic BPE    Advanced BPE (Dropout)
-----------------------------------------------------------------
Round-trip correct            PASS         PASS
Compression ratio             0.2251       0.2251
Fertility tokens/word         1.1855       1.1855
Unique token types used       3778         3778
UNK rate                      N/A          0.0000
Frag. on 500 rare words       3.0220       3.1880
Train time (s)                269.85       283.83
Encode time (s)               89.70        86.60
```

Round-trip passes for both which is the basic correctness check.

Compression ratio and fertility are identical because both models use the same merge table. Deterministic encoding at p = 0 always produces the same result, so these numbers will never differ between the two at inference.

UNK rate is 0.0 for the dropout model, confirming the byte-level fallback handles everything.

Fragmentation on rare words is slightly higher for the dropout model (3.188 vs 3.022). This is because some merges get skipped during the stochastic encoding pass used for this measurement. At inference with p = 0 this difference goes away entirely.

Training takes about 14 seconds longer which is a reasonable cost. Encode time is basically the same, within noise.

The honest limitation here is that the real benefit of dropout is not visible in any of these numbers. It only shows up if you train a language model and compare perplexity on held-out text, which is outside the scope of this task.

---

## 5. Related Work

Sennrich, Haddow & Birch (2016) introduced BPE for NLP. The original implementation is fully deterministic with no stochasticity in encoding.

Provilkov, Emelianenko & Voita (2020) introduced BPE-Dropout, which is the direct basis for this part. They show stochastic segmentation during training improves translation quality and morphological robustness, especially in low-resource settings.

Kudo (2018) proposed Subword Regularization, a similar idea but applied to a unigram language model tokenizer instead of BPE. It samples segmentations from a probability distribution rather than randomly dropping merges.

Kudo & Richardson (2018) built SentencePiece, which includes both BPE and unigram tokenization with subword regularization built in.

Bostrom & Durrett (2020) showed that vanilla BPE is suboptimal for language model pretraining and that stochastic alternatives address some of the identified problems.

---

## 6. Summary

The problem with vanilla BPE is that it always produces one fixed segmentation per word, which limits what the model can learn about subword structure. BPE-Dropout fixes this by randomly skipping merge rules during encoding, so the model sees varied splits of the same words during training. The change is three lines in the encode loop and nothing else. At inference you set dropout to zero and get normal deterministic BPE back. The tokenizer-level results are mostly identical between the two because they share the same merge table — the real benefit would only appear in a downstream model evaluation.

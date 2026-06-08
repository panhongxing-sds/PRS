# Process Dynamics: Early Warning & Localization (not pointwise prediction)

**Framing:** Token-side competition provides an **early, local diagnostic footprint** before 
answer-level fragmentation — not a strong stepwise classifier.

Primary datasets: Minerva, MATH-500, DeepScaler (GSM8K omitted as atypical).

## 1. Event-study aligned at τ_A (first A spike / cluster switch)

Mean T, ΔT, A, ΔA at offsets relative to fragmentation event.

### deepscaler (n=192 with A-event)

| offset | mean T | mean ΔT | mean A | mean ΔA |
|-------:|-------:|--------:|-------:|--------:|
| -3 | 0.900 | 0.017 | 0.008 | -0.002 |
| -2 | 0.888 | -0.029 | 0.007 | -0.003 |
| -1 | 0.882 | -0.024 | 0.005 | -0.001 |
| +0 | 0.814 | -0.082 | 0.480 | 0.267 |
| +1 | 0.776 | -0.039 | 0.276 | -0.204 |
| +2 | 0.768 | -0.008 | 0.239 | -0.037 |

## 2. Raw T_k vs ΔT_k (secondary; pointwise prediction weak)

| Dataset | T→wrong | ΔT→wrong | ΔT→future frag | mean lead (wrong) | frac T before A |
|---------|--------:|---------:|---------------:|------------------:|----------------:|
| deepscaler | 0.507 | 0.502 | 0.458 | 3.80 | 62.6% |

## 3. Spike taxonomy (process-defined, no label leakage)

| Dataset | Category | n | wrong rate | pre-frag rate |
|---------|----------|--:|-----------:|--------------:|
| deepscaler | pre_frag_persistent | 92 | 81.5% | 100.0% |
| deepscaler | persistent | 42 | 71.4% | 0.0% |
| deepscaler | pre_fragmentation | 3 | 66.7% | 100.0% |

## 4. Spike location at first T spike

| Dataset | Location | n | wrong rate | frac T before A |
|---------|----------|--:|-----------:|----------------:|
| deepscaler | numeric | 66 | 69.7% | 95.5% |
| deepscaler | operator | 31 | 74.2% | 0.0% |
| deepscaler | other_math | 22 | 100.0% | 45.5% |
| deepscaler | equation | 18 | 88.9% | 77.8% |

## Interpretation (recommended claims)

1. **Mechanistic:** T rises before A at τ_A−1 in event-study (visual pre-fragmentation bump).
2. **Localization:** equation/numeric spikes more often precede A than generic math tokens.
3. **Action:** pre-fragmentation / persistent spikes → candidate **intervention triggers**, not final classifiers.
4. **Do not claim:** raw T_k or ΔT_k as strong pointwise predictors (AUROC may stay ~0.5).

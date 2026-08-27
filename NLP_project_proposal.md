**Project Proposal - Students: Ilana Kolody, Itai Mor, Shai Habi**

**Project title:** Does AI Neglect Rare Science? Long-Tail Topic Classification of Astronomy Papers with Small Encoders vs. Quantized Generative LLMs  
**Research question**  
Astronomy papers are tagged with concepts from a large, expert-built controlled vocabulary (the Unified Astronomy Thesaurus, UAT; \~2,300 concepts), in which a few topics are very common and most are rare. We ask:  
Can parameter-efficient fine-tuning (LoRA / QLoRA) let a large generative model (Gemma) match or beat a small specialized encoder (SciBERT) at multi-label topic classification of astronomy papers? We are especially interested in the rare, long-tail topics where models tend to struggle most, and in whether the large model's extra compute is justified. We answer this by measuring each model's performance across the label frequency spectrum (common / medium / rare), so the behavior on rare topics is visible rather than hidden inside an overall average.  
The question is framed to be informative either way: if a long-tail gap exists, we characterize it and test whether each method narrows it. If it is unexpectedly small, that itself is a reportable finding.

## **Motivation - and why astronomy specifically**

Long-tail multi-label classification and parameter efficient fine-tuning are well studied in general NLP.  
Prior examples include balancing-loss methods for long-tailed classification (Huang et al., EMNLP 2021) and transformer-based extreme multi-label classification (Chang et al., X-Transformer, KDD 2020).So why study this in astronomy rather than treat it as solved? Two reasons make the setting distinct:

**1. Real stakes on an expert-built vocabulary.** The UAT isn't a random set of tags, it's a formal thesaurus put together by astronomers, and NASA's SciX/ADS uses it to organize the astronomy literature. So if a model keeps missing the rare topics, real niche research becomes harder to find. That's a concrete, real-world cost that generic sentiment or news datasets just don't have.

**2. The domain gap is unusually large.** Astronomy abstracts are dense with jargon, object designations, and formula like notation. That makes the "does a general generative LLM match a domain-pretrained encoder?" comparison more informative here, since domain specialization is exactly what stresses the models.

## **What is known and what we add**

Several things are already known. Domain-pretrained encoders like SciBERT are strong on scientific text, and LoRA/QLoRA make fine-tuning cheap. Long-tail degradation is also well documented in extreme multi-label classification: rare labels are where strong models collapse (Chang et al., KDD 2020), and balancing methods help but don't fully solve it (Huang et al., EMNLP 2021). Per-frequency-band (head/medium/tail) evaluation is established too (Huang et al., EMNLP 2021), including on a specialized scientific vocabulary. So the breakdown itself isn't our contribution.

What we add is the comparison. We put a small domain-pretrained encoder (SciBERT) against a quantized generative LLM (Gemma + QLoRA) on the astronomy UAT, and we look at results across the label-frequency spectrum. The questions are simple: which one handles rare topics better, and is the large model's extra compute worth it? To our knowledge, this pairing hasn't been reported.

**Method (brief)**

-   **Data:** adsabs/SciX\_UAT\_keywords (HuggingFace; MIT license; \~21.7K papers; title + abstract + multi-label UAT tags). We keep labels that appear at least *k* times, a standard minimum frequency threshold. This ensures every retained label has enough examples to be trained and evaluated fairly. The rare-but-learnable labels, which are the focus of the study, will still remain.

-   **Baselines:** a majority/frequency predictor and a TF-IDF + Logistic Regression classifier.

-   **Models compared:** (a) SciBERT with full fine-tuning, (b) SciBERT with LoRA, (c) a small quantized Gemma (4-bit) fine-tuned with QLoRA, producing labels generatively.  
    All trained and evaluated under an identical protocol.

-   **Evaluation:** We report precision/recall/F1 per frequency band (common / medium / rare), so the long-tail behavior is visible. We also report compute cost (trainable parameters, GPU-hours), which makes the small-vs-large trade-off explicit. And if results are not meaningful, we run a standard sanity check: overfitting a small subset to confirm the pipeline is bug-free.

## **Anchor papers (NLP/ML venues, post-2018)**

**1. SciBERT** - Beltagy, Lo & Cohan, EMNLP 2019 (base encoder).

**2. LoRA** - Hu et al., ICLR 2022 (parameter-efficient fine-tuning).

**3. QLoRA** - Dettmers et al., NeurIPS 2023 (4-bit quantized fine-tuning for the Gemma arm).

## **Assumptions and resource requirements**

-   **Data:** fully open (MIT), no access gating, \~19 MB - already in hand.

-   **Compute:** all models are small (SciBERT \~110M; Gemma 4B in 4-bit \~3 GB). Free Colab (T4) suffices.  
    We will also use the TAU Slurm studentkillable partition for the Gemma arm and repeated runs.  
    Storage needs are minimal (we avoid frequent checkpointing).

-   **No external API credits or special hardware required.** The SciBERT-vs-Gemma comparison is our core contribution, so our Plan B scales it down rather than dropping it. If the full Gemma + QLoRA arm proves too costly in time, we fall back to a smaller Gemma (1B instead of 4B) or fewer runs. This keeps the comparison intact, just cheaper.

*Note: this proposal runs over one page. We kept the extra length deliberately to address the points raised in your email, situating the work against prior research and clarifying why the astronomy setting is worth studying.*

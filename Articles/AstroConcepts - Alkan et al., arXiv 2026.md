# AstroConcepts: A Large-Scale Multi-Label Classification Corpus for Astrophysics

**Alkan, Atilla Kaan; Grezes, Felix; Blanco-Cuaresma, Sergi; Bartlett, Jennifer Lynn; Chivvis, Daniel;
Kelbert, Anna; Lockhart, Kelly; Accomazzi, Alberto**

Harvard-Smithsonian Center for Astrophysics, Cambridge, MA, USA

**arXiv:2604.02156v1 [cs.CL], 2 April 2026** — preprint, not peer-reviewed.

> Converted from PDF for reference. Text reconstructed from character positions, so minor spacing or
> hyphenation artefacts may remain. Check the PDF before quoting verbatim in the paper.

---


<!-- page 1 -->

AstroConcepts: A Large-Scale Multi-Label Classification Corpus for
Astrophysics
Atilla Kaan Alkan1, Felix Grezes1, Sergi Blanco-Cuaresma1,2,
Jennifer Lynn Bartlett1, Daniel Chivvis1, Anna Kelbert1,
Kelly Lockhart1, Alberto Accomazzi1
1Harvard-Smithsonian Center for Astrophysics, Cambridge, MA, USA
2Faculty of Psychology, UniDistance Suisse, Brig, Switzerland
{atilla.alkan, felix.grezes, sblancocuaresma, jennifer.bartlett, daniel.chivvis, anna.kelbert, kelly.lockhart, alberto.accomazzi}@cfa.harvard.edu
Abstract
6202
Scientific multi-label text classification suffers from extreme class imbalance, where specialized terminology exhibits severe power-law distributions that challenge standard classification approaches. Existing scientific corpora lack comprehensive controlled vocabularies, focusing instead on broad categories and limiting systematic study of extreme imbalance. We introduce AstroConcepts, a corpus of English abstracts from 21,702 published astrophysics papers, rpA labeled with 2,367 concepts from the Unified Astronomy Thesaurus. The corpus exhibits severe label imbalance, with 76% of concepts having fewer than 50 training examples. By releasing this resource, we enable systematic study of extreme class imbalance in scientific domains and establish strong baselines across traditional, neural, and
2 vocabulary-constrained LLM methods. Our evaluation reveals three key patterns that provide new insights into scientific text classification. First, vocabulary-constrained LLMs achieve competitive performance relative to domain-adapted
]LC.sc[ models in astrophysics classification, suggesting a potential for parameter-efficient approaches. Second, domain adaptation yields relatively larger improvements for rare, specialized terminology, although absolute performance remains limited across all methods. Third, we propose frequency-stratified evaluation to reveal performance patterns that are hidden by aggregate scores, thereby making robustness assessment central to scientific multi-label evaluation.
These results offer actionable insights for scientific NLP and establish benchmarks for research on extreme imbalance.
1v65120.4062:viXra
Keywords: Multi-label Text Classification, Scientific Document Classification, Extreme Label Imbalance
1. Introduction The astrophysics literature exemplifies these challenges while providing an ideal testbed for systematic investigation. The Unified Astronomy TheScientific multi-label text classification poses sigsaurus (UAT) (Accomazzi et al., 2014) organizes nificant challenges due to extreme class imbal2,367 concepts across 11 hierarchical levels, creance, with specialized terminology exhibiting seating a comprehensive controlled vocabulary that vere power-law distributions that challenge stanexhibits natural power-law distributions characterisdard classification methods (Liu et al., 2023). While tic of scientific domains. imbalanced distributions are common in scientific
To provide the community with essential redomains, existing corpora (Giles et al., 1998; Mcsources for investigating extreme multi-label classiCallum et al., 2000; Yang et al., 2018) typically fication in scientific domains, we introduce Astroprovide limited coverage of controlled vocabulary,
Concepts, a corpus of 21,702 astrophysics papers focus on broad disciplinary categories, or operate labeled with the complete UAT vocabulary (2,367 at scales that hinder systematic methodological inconcepts). This resource enables systematic invesvestigation of extreme imbalance scenarios. This tigation of three fundamental research questions: limitation is particularly problematic for comprehensive scientific text classification, where controlled • RQ1 How do traditional methods, supervised vocabularies organize thousands of specialized neural models, and vocabulary-constrained concepts with naturally occurring power-law dis- LLMs compare for handling extreme label imtributions. Existing datasets that provide controlled- balance in scientific classification? vocabulary coverage either operate at prohibitive
• RQ2 Does domain-specific pretraining provide computational scales (Toney and Dunham, 2022) uniform benefits across frequency bins, or do or focus on limited subsets of domain terminology, improvements concentrate in specific regions thereby preventing systematic investigation of how of the label distribution? different approaches address the fundamental challenge of learning from severely imbalanced special- • RQ3 Can vocabulary-constrained LLMs ized terminologies. achieve competitive performance with

<!-- page 2 -->

domain-adapted models in astrophysics 1987), while legal document classification inclassification? troduced a hierarchical structure through EURLex (Loza Mencía and Fürnkranz, 2010), though
Our main contributions are: annotations include complete paths rather than the flat terminal concepts typical in practice. Extreme
1. AstroConcepts1 provides the community multi-label benchmarks (Amazon-670K, Wiki-500K) with the first tractable-scale corpus, enabling scale to hundreds of thousands of labels but priorisystematic investigation of extreme multi-label tize computational efficiency over domain-specific classification with comprehensive controlledcontrolled vocabularies. vocabulary coverage.
Scientific classification began with computer sci2. Systematic comparison across traditional, ence papers: Giles et al. (1998) used six broad neural, and vocabulary-constrained LLM apcategories, McCallum et al. (2000) expanded to 70 proaches reveals competitive performance for categories with a 3-level hierarchy, and later work parameter-efficient methods, opening new reincorporated controlled vocabularies (Santos and search directions for scientific NLP.
Rodrigues, 2009; Kowsari et al., 2017; Yang et al.,
2018), although these remained limited in scope.
3. Introduction of frequency-stratified evaluation
Recent efforts leverage Microsoft Academic Graph: framework with robustness metrics that reveal
Cohan et al. (2020) created embeddings across 19 performance patterns invisible in aggregate fields, Sadat and Caragea (2022) scaled to 186K scores, providing essential tools for extreme papers with 6-level hierarchy, and Toney and Dunmulti-label assessment. ham (2022) used 180–220M papers, though com4. Demonstration that domain adaptation im- putational demands limit systematic experimentaprovements concentrate on rare specialized tion. While MAG provides broad coverage, it spans terminology, with implications for training multiple disciplines rather than offering deep dostrategies in scientific applications. main specialization.
Table 1 puts AstroConcepts in context of ex5. Comprehensive baseline establishment that isting resources. General benchmarks lack conprovides essential benchmarks for scientific trolled vocabularies, whereas scientific corpora eimulti-label classification while revealing funther use ad hoc categories, span multiple discidamental properties of extreme imbalance in plines without deep specialization, or reach a prospecialized domains. hibitive scale. AstroConcepts uniquely combines tractable scale (21K), deep hierarchy (11 levels),
The remainder of this paper is structured as foland domain-specific controlled vocabulary (UAT), lows. Section 2 reviews related work in scientific enabling comprehensive evaluation of hierarchymulti-label classification and positions AstroConaware methods and few-shot learning in a realistic cepts within the existing landscape of scientific corscientific classification scenario. pora. Section 3 describes the corpus construction methodology, annotation procedures, and key characteristics of the resulting dataset. Section 4 details
3. The AstroConcepts Corpus our experimental setup, including baseline methods, evaluation metrics, and a frequency-stratified
AstroConcepts exhibits characteristics that make analysis framework. Section 5 presents results scientific multi-label classification challenging: a across all methods, revealing key patterns in overhierarchical controlled vocabulary (UAT) with flat all performance and frequency-specific behaviors. expert annotations, severe label imbalance, and
Section 6 discusses the broader implications of our domain-specific terminology that requires specialfindings for scientific NLP, methodological contribuized language understanding. This section detions, and limitations of the current work. Section 7 scribes the data collection process, the UAT taxconcludes with a summary of key insights and dionomy structure, the annotation methodology, and rections for future research. the comprehensive corpus statistics.
2. Scientific Multi-label Classification
3.1. Source Data
Benchmarks
We collected 21,702 abstracts from published
Multi-label text classification spans diverse do- English-language papers indexed by the NASAmains and scales. Early work established eval- funded Science Explorer (SciX; Bartlett et al. uation protocols with news categorization (Lewis, (2025)), an expansion of the Astrophysics Data
System (ADS; Accomazzi et al. (2015)) to cover
1https://huggingface.co/datasets/ all NASA science disciplines. Building on the ADS legacy, SciX is the primary bibliographic database adsabs/SciX_UAT_keywords

<!-- page 3 -->

Corpus Docs Labels Hierarchy Supervision Vocabulary Domain
General Domain
Loza Mencía and Fürnkranz (2010) 19K 7.2K 2-level Hierarchical EuroVoc Legal
Lewis (1987) 21K 90 None Flat Ad-hoc News
RCV1 800K 103 4-level Flat Ad-hoc News
Scientific Domain
Giles et al. (1998) 3K 6 None Flat Ad-hoc CS
Santos and Rodrigues (2009) 15K 92 2-level Mixed ACM CS
AstroConcepts 21K 2.3K 11-level Flat UAT Astrophys
Cohan et al. (2020) 25K 19 1-level Flat MAG Multi-sci
Kowsari et al. (2017) 47K 134 2-level Hierarchical WoS CS+Med
McCallum et al. (2000) 53K 70 3-level Hierarchical Ad-hoc CS
Yang et al. (2018) 55.8K 54 2-level Flat Ad-hoc CS
Sadat and Caragea (2022) 186K 1.2K 6-level Mixed MAG Multi-sci
Toney and Dunham (2022) 180M 313 2-level Flat MAG Multi-sci
Table 1: Multi-label classification benchmarks sorted by corpus size. AstroConcepts provides tractable scale (21K documents) with deep hierarchical structure (11 levels) and domain-specific controlled vocabulary (UAT), enabling comprehensive experimentation on astrophysics literature classification. for astronomy and astrophysics. It indexes papers an eleven-level hierarchical structure forming a difrom approximately 8,000 refereed journals, includ- rected acyclic graph (DAG). Concepts range from ing the Astrophysical Journal (ApJ). Starting in broad top-level categories such as cosmology or ob2018, journal editors began requiring or encour- servational astronomy, to highly specific concepts aging UAT concept assignment during submission such as the Kreutz group. Each concept includes a to promote standardized concept usage in astron- unique identifier, a canonical designation, alternaomy, resulting in a growing corpus of controlled- tive names (synonyms, acronyms), and an optional vocabulary annotations. textual definition (scope note), and explicit relationWe selected papers meeting the following crite- ships to parents, children, and related concepts. ria: (1) published in journals where authors assign The UAT follows a polyhierarchical structure in
UAT concepts during submission, ensuring con- which concepts can have multiple parents, creating trolled standard concept annotation, (2) at least one a DAG topology in which nodes (concepts) can be
UAT concept assigned, and (3) English-language reached through multiple paths without forming cyabstracts. The temporal scope covers 2018-2023, cles. For example, appears
Stellar atmosphere reflecting the data available during corpus construc- under both and Spectroscopy, as studies
Stars tion. Future work could extend coverage to more of stellar atmospheres involve both stellar physics recent publications. This process yielded 21,702 and spectroscopic methods. This DAG structure documents. reflects the multidisciplinary nature of astronomical research, where concepts naturally belong to multiple semantic categories simultaneously.
3.2. The Unified Astronomy Thesaurus
The UAT (Accomazzi et al., 2014) is a community3.3. Concept Assignment Process maintained controlled vocabulary for astronomical literature, owned and openly licensed by the
Amer- Labels in AstroConcepts come from the standard
Society. It follows the Simple ican Astronomical publishing process where authors assign UAT conKnowledge Organization System (SKOS; Miles and cepts to their manuscripts during submission. AuBechhofer (2009)) standard and incorporates terms thors typically select 4 concepts per paper (see Tafrom earlier astronomy thesauri. Astronomy librarble 2) using the journal’s submission system. They ians and domain experts have contributed to its choose specific concepts without marking hierardevelopment and maintenance. Released in 2017 chical paths. For example, selecting
Exoplanet and subsequently adopted by major journals and does not require selecting its broader atmospheres
SciX, the UAT provides standardized terminology categories, such as astronomy. This creExoplanet for indexing and retrieval of astronomy content. ates an interesting challenge: systems must predict
The UAT version 5.1.02 used in this work conspecific concepts from a hierarchical vocabulary tains 2,367 astronomical concepts organized in using only flat annotations. The approach has clear advantages: we collect high-quality labels from do2https://github.com/astrothesaurus/ main experts who know their work best, and the standardized UAT vocabulary ensures consistency.
UAT/tree/v.5.0.0

<!-- page 4 -->

However, authors naturally focus on what they con- label scenarios. Figure 1 shows the label frequency sider most important rather than providing com- distribution on a log-log scale, revealing a powerprehensive coverage. This means some relevant law pattern. The most frequent label (Galaxy evoconcepts might be missing, creating a realistic yet lution) appears in 1,106 abstracts (5.1%), while the challenging evaluation scenario that mirrors real- median label appears in only 12 abstracts. We fit world conditions in which systems operate with an a power law r−α where is rank and f (r) ∝ r f (r) incomplete set of annotations. is frequency, obtaining α = 1.50 with R2 = 0.825.
3.4. Overall Statistics
Observed
105
Power law fit: =1.50
Table 2 presents overall corpus statistics. Astro104
Concepts contains 21,702 abstracts with 93,547 ycneuqerF total label assignments, averaging 4.31 labels per
103 abstract. The assigned label space comprises
102
1,864 unique UAT concepts (92% of the UAT vocabulary), indicating that the selected literature ab101 stracts span the conceptual space defined by UAT.
100
100 101 102 103
Statistic Value
Label Rank
Documents
Figure 1: Label frequency distribution (log-log
Total abstracts 21,702 scale) and fitted power-law function with exponent
Labels
α = 1.50 (R2 = 0.825). The long tail contains 76%
Total assignments 93,547 of labels with fewer than 50 occurrences each, creAssigned unique labels 1,864 ating a severe class imbalance characteristic of
Per abstract (mean/median) 4.31 / 4
Per abstract (range) 1–12 scientific multi-label classification.
Text
The exponent indicates a moderately
α = 1.50
Length in words (mean/median) 211.2 / 223 steep long-tail distribution. This suggests that AsLength (range) 18–462 troConcepts presents a substantial but not exVocabulary size 163,326 treme class imbalance compared to other scientific classification scenarios. The moderate slope
Table 2: Overall statistics for AstroConcepts. means mid-frequency (torso) labels retain more training examples than in steeper distributions; by
Abstracts average 211 words (median: 223), typirank 100, labels still average 58 examples, whereas cal for scientific abstracts and compatible with stanin datasets with 2.0, rank-100 labels would dard transformer context windows (512 tokens). α = have only 25 examples. Nevertheless, a severe
The distribution is approximately normal with slight imbalance remains: 76% of labels occur fewer than positive skew toward longer abstracts; 94% fit
50 times. within 512 tokens. We retain abstracts up to 462
We partition labels into three frequency bins (see words to capture the full range of scientific writing
Table 3). Head labels (frequency 500): 17 lastyles without truncation artifacts. > bels (0.9% of vocabulary) covering 12,288 assignThe distribution of labels per abstract (mean: ments (13.1% of total). These represent core con4.31, median: 4, range: 1–12) reflects the mulcepts studied extensively across astrophysics subtifaceted nature of astrophysics research. Most fields. Torso labels (50 frequency 500): 429 laabstracts (70%) receive 2–5 labels, with single- ≤ ≤ bels (23.0%) covering 62,808 assignments (67.1%). label papers typically representing narrowly foThese represent moderately common research topcused studies and papers with 6+ labels (15%) ics with sufficient training examples for standard covering interdisciplinary or methodologically disupervised learning. Tail labels (frequency 50): verse work. This moderate label density exceeds <
1,418 labels (76.1%) covering 18,451 assignments general multi-label benchmarks such as Reuters-
(19.7%). These represent specialized phenomena,
21578 (Lewis, 1987), which averages 1.2 labels emerging research areas, and niche methodoloper document (Huang et al., 2021), underscoring gies with limited training examples, creating the the conceptual complexity inherent in scientific literature. zero-shot challenge we investigate in Section 4.
This distribution creates the multi-level challenge characteristic of extreme multi-label classification:
3.5. Concept Frequency Distribution head labels are easily learned from abundant exA critical characteristic of AstroConcepts is se- amples (hundreds per label), torso labels require vere label imbalance, typical of real-world multi- careful modeling to generalize from moderate data

**[Table on page 4]**

| Observed 105 Power law fit: =1.50 104 ycneuqerF 103 102 101 100 100 101 102 103 Label Rank |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | Ob | ser | ve | d |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | Po | wer | la | w | f | it | : | =1.5 | 0 |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |


<!-- page 5 -->

Bin # Labels % # Assign. % Depth Count %
Head (>500) 17 0.9 12,288 13.1
1 106 4.5
Torso (50–500) 429 23.0 62,808 67.1
Tail (<50) 1,418 76.1 18,451 19.7 2 106 4.5
3 405 17.1
Total 1,864 100.0 93,547 100.0
4 680 28.7
Table 3: Label frequency distribution across bins.
5 554 23.4
The long tail contains 76% of labels but only 20%
6 341 14.4 of assignments, creating a severe class imbalance
7 110 4.6 characteristic of scientific multi-label classification.
8 36 1.5
9 13 0.5
10 11 0.5
(50–500 examples), and tail labels present a few11 5 0.2 shot scenario (fewer than 50 examples, median:
12) where standard supervised methods struggle.
Table 4: Distribution of UAT concept depths in AsThe most frequent labels span major astrotroConcepts showing annotation preference patphysics subfields: extragalactic astronomy (Galaxy terns across taxonomic levels. evolution, nuclei), stellar physics
Active galactic
(Star formation, stars), planetary science
Neutron
(Exoplanets, atmospheres), and obserExoplanet from unrelated subtrees show 2.3× higher average vational methods (Spectroscopy,
Astronomy data PMI than parent-child pairs, indicating that authors analysis). No single subfield dominates the top-20 typically choose concepts spanning multiple taxlabels, confirming that AstroConcepts captures onomic branches rather than selecting both genthe breadth of modern astrophysics research rather eral and specific terms from the same hierarchy. than concentrating on narrow phenomena. This diThis cross-branch annotation behavior represents versity is important for multi-label classification rea challenge for classification systems, which must search: the corpus contains both well-represented learn to predict conceptually diverse label combicore concepts and sparse specialized topics, ennations spanning the entire taxonomic structure. abling evaluation across the full spectrum of label frequencies.
4. Experiments
3.6. Concept Specificity Patterns
4.1. Experimental Setup
Table 4 reveals that authors prefer moderately speTask Formulation We formulate astrophysics cific concepts (peaking at level 4). Both very genconcept classification as a multi-label task where, eral concepts (levels 1-2) and highly specialized given an abstract concatenated with its title, the x terminology (levels 6+) are underrepresented relgoal is to predict a subset of relevant UAT
Y ⊆ L ative to mid-level concepts. This concentration concepts from the complete label space of 2,367
L around moderate specificity creates additional challabels. We use title+abstract concatenation as inlenges for classification systems beyond frequency put text to provide models with maximum available imbalance. Models must handle vocabularies in semantic information for concept prediction. which training examples cluster around mid-level concepts, with limited examples at both ends of the
Data Partitioning We split the corpus into train taxonomic spectrum, requiring systems to predict
(18,677 abstracts, 85%) and test (3,025 abstracts, across varying specificity levels with highly imbal15%) sets using label-aware stratification. Labels anced training signals. appearing ≥15 times are stratified to achieve approximately 85/15 distribution per label, while la3.7. Concept Co-occurrence Patterns bels with <15 occurrences are placed entirely in
We analyzed how concepts co-occur in our cor- the training set to maximize training signal. pus. We computed pointwise mutual information (PMI) for concept pairs, where PMI(ℓ , ℓ ) =
Evaluation Metrics Following standard practice i j
P (ℓi,ℓj) log measures whether two concepts ap- in multi-label classification (Zhang and Zhou, 2014;
P (ℓi)P (ℓj) pear together more frequently than expected by Tsoumakas and Katakis, 2010), we evaluate uschance. Positive PMI indicates stronger-than- ing Macro-F1, which averages per-label F1 scores expected co-occurrence, while negative PMI sug- and is essential for imbalanced settings (Wu et al., gests mutual exclusivity. Focusing on pairs with 2020). For ranking evaluation, we use Precipositive PMI and sufficient co-occurrence (≥ 10 sion/Recall at (P @k, R@k) with 5}, chok k ∈ {1, 3, abstracts), we found that authors rarely select hi- sen to align with the average number of assigned erarchically related concepts together. Concepts concepts per paper (4.31, see Table 2).

<!-- page 6 -->

4.2. Baseline Approaches perform a grid search over learning rates
{2e − and epochs to find
5, 3e − 5, 5e − 5} {1, 2, 3, 5, 8, 10}
We establish baselines across multiple paradigms optimal hyperparameter configurations. Detailed to understand effective modeling approaches results for each configuration are provided in the for astrophysics concept classification, organized appendix. Table 5 reports only the best-performing by increasing complexity: non-parametric methconfiguration (lr=2e-5, epochs=8) for readability. ods, supervised neural models, and vocabularyconstrained LLMs.
4.2.3. Vocabulary-Constrained LLMs
4.2.1. Non-Parametric Methods
LLMs demonstrate strong capabilities on multilabel text classification tasks (Zhou et al., 2024;
Rule-based Matching We implement lexical
Tabatabaei et al., 2025) but face challenges with matching based on the assumption that if a UAT large label spaces. Prompting an LLM directly to concept is explicitly mentioned in the text, it should select from all 2,367 UAT concepts is infeasible due be assigned as a label. The method searches to context length limitations and label complexity. for exact string matches of UAT concept names
Preliminary experiments on a small subset yielded within the title and abstract. For each concept, we very poor results when using the complete label list, check for its canonical designation and optionally as the model hallucinated concepts or became overinclude synonyms and abbreviations from the UAT whelmed by the extensive vocabulary. We therefore taxonomy (e.g., searching for both active galactic implement a two-stage approach: (1) use our best and AGN). We evaluate two variants: Rulenuclei supervised model (astroBERT; see Table 5) to genbased uses only canonical names, while Rulew/o var erate top-50 candidate labels for each abstract, (2) based includes all alternative forms provided w/ var prompt (DeepSeek-AI
DeepSeek-V3-reasoner in the UAT. et al., 2024) via its API4 to select relevant concepts from these candidates. We chose the top-50 threshk-Nearest Neighbors This approach assumes old based on analysis showing that astroBERT’s abstracts with high contextual similarity should top-50 predictions covered approximately 82% of share similar concepts. We encode titles and ground-truth labels (see Figure 3 in the appendix 9), abstracts using three embedding models: asproviding good coverage while maintaining mantroBERT (Grezes et al., 2024) (adapted specifically ageable prompt length. We evaluate
DeepSeekfor astrophysics texts), INDUS (Bhattacharjee et al., using the prompt shown in Figure 2
V3-reasoner
2024) (a scientific language model covering astroof the appendix 9. This approach constrains the physics, earth science, and general physics), and model to valid UAT terminology while leveraging its
Qwen3-Embedding-8B3
(chosen for strong sensemantic understanding to select the most relevant tence similarity performance). For each abstract concepts from the candidate set. from the test set, we retrieve nearest training k neighbors using cosine similarity, then predict the
5. Results and Analysis most frequent labels among them. We perform a grid search over and all embedk ∈ {5, 10, 20, 50}
This section presents and analyzes results across ding models, with detailed results provided in the all methods, revealing key insights about domain appendix. Table 5 reports only the best-performing adaptation, frequency effects, and the comparaconfiguration (embedding model and value) for k tive strengths of different learning paradigms for readability. extreme multi-label scientific classification.
4.2.2. Supervised Neural Models
5.1. Overall Performance and Domain
To investigate the effects of domain adaptation, we
Impact fine-tune three transformer models representing
Table 5 presents comprehensive results across different levels of domain specialization: BERT (Deall methods, revealing fundamental insights about vlin et al., 2019) (general-purpose), SciBERT (Beltlearning paradigms for extreme multi-label scientific agy et al., 2019) (scientific domains), and asclassification. troBERT (Grezes et al., 2024). We add classificaOur evaluation reveals a progression of insights tion heads on [CLS] representations and fine-tune across methodological paradigms. Simple ruleend-to-end with max_length=512, batch_size=8, based matching achieves reasonable precision and AdamW optimizer. Due to computational re-
(0.229) but faces fundamental limitations: string source constraints, we were unable to include matching of concept mentions may not reflect the
Qwen models in the fine-tuning experiments. We core research focus. Common terms like "photon"
3https://huggingface.co/Qwen/
4https://api-docs.deepseek.com/
Qwen3-Embedding-8B

<!-- page 7 -->

Overall Ranking Precision Ranking Recall
Method Precision Recall F P@1 P@3 P@5 R@1 R@3 R@5
1
Non-parametric
Rule-based 0.2290 0.2130 0.1950
‡ ‡ ‡ ‡ ‡ ‡ w/o var
Rule-based 0.2170 0.2730 0.2140 ‡ ‡ ‡ ‡ ‡ ‡ w var k-NN 0.1165 0.7509 0.2017 0.6125 0.4607 0.3698 0.1398 0.3155 0.4221
Supervised neural models
BERT 0.1784 0.2273 0.1880 0.2975 0.2187 0.1784 0.0810 0.1730 0.2273
SciBERT 0.2020 0.2563 0.2127 0.3213 0.2409 0.2020 0.0872 0.1864 0.2563 astroBERT 0.3068 0.3905 0.3243 0.4909 0.3733 0.3068 0.1360 0.2933 0.3905
Zero-shot prompting
Deepseek 0.2891 0.6322 0.3770 0.6502 0.4930 0.4017 0.1815 0.3837 0.5050
Table 5: Overall performance on AstroConcepts test set. Best results shown in bold. ‡Not applicable for rule-based methods. appear across diverse papers, while implicit lan- sights into handling extreme imbalance. First, doguage and incomplete variant coverage constrain main adaptation provides asymmetric benefits: aseffectiveness. troBERT shows modest Head improvements over
Moving to similarity-based approaches, k-NN SciBERT (0.193 0.216) but larger relative gains
→ achieves exceptional recall (0.751) but poor preci- for Tail concepts (0.023 0.081), though ab-
→ sion (0.117). Crucially, domain-adapted astroBERT solute performance remains limited across tradiembeddings outperform general models like Qwen, tional approaches. Second, all methods exhibit better capturing subtle concept distinctions essen- a consistent "torso peak" where mid-frequency tial for astrophysics. However, performance de- concepts achieve optimal performance. This patgrades beyond neighbors as noise from tern suggests fundamental properties of scienk = 10 irrelevant papers accumulates. tific vocabulary learning, in which models balance sufficient training signal with complexity issues
The most striking finding emerges with at frequency extremes. Most significantly, the vocabulary-constrained DeepSeek, which outvocabulary-constrained approach achieves supeperforms even the best domain-adapted model
(astroBERT) by 16% in F score (0.377 vs 0.324). rior tail performance (F : 0.198 vs 0.081 for as1
1
This demonstrates that domain expertise can troBERT), demonstrating that domain expertise can be effectively incorporated through vocabulary be effectively incorporated through structured conconstraints rather than solely through model straints rather than parameter fine-tuning alone. parameters. Meanwhile, supervised domain This hybrid approach, combining general language adaptation shows clear value: astroBERT sub- understanding with domain-specific vocabulary stantially outperforms SciBERT (0.324 vs 0.213) guidance, proves particularly effective for rare conwith improvements concentrated in precision and cepts. Finally, we introduce frequency robustness confident prediction rather than comprehensive (∆ = Head F - Tail F ) as a critical evaluation di1 1 recall. mension. The constrained approach achieves 3× better robustness than SciBERT (∆ = 0.045 vs
Together, these results reveal that effective sci0.170), indicating that architectural choices and entific text classification benefits from hybrid apconstraint mechanisms matter more for handling proaches combining general language understandfrequency imbalance than domain specialization ing with structured domain knowledge, opening alone. promising directions for parameter-efficient scientific NLP.
6. Discussion
5.2. The Long-Tail Challenge
Our systematic evaluation reveals fundamental insights into extreme multi-label classification in sciThe extreme label distribution in AstroConcepts entific domains while establishing new evaluation
(78% of concepts have < 50 examples) enables the paradigms that advance understanding beyond exsystematic analysis of long-tail performance patisting benchmarks. terns. Table 6 presents frequency-stratified results across Head (> 500 examples), Torso (50–500), and Tail (< 50) concepts, revealing insights that Addressing the Research Questions Our findaggregate metrics cannot capture. ings provide clear answers to the three research
Three patterns emerge, providing useful in- questions posed. Regarding RQ1 (handling ex-

<!-- page 8 -->

F Head Torso Tail
1
Method Head Torso Tail P@3 R@3 P@3 R@3 P@3 R@3 ∆∗
BERT 0.180 0.167 0.021 0.084 0.205 0.168 0.183 0.012 0.024 0.159
SciBERT 0.193 0.197 0.023 0.084 0.211 0.199 0.212 0.015 0.026 0.170 astroBERT 0.216 0.312 0.081 0.092 0.230 0.311 0.333 0.046 0.088 0.135
DeepSeek 0.243 0.376 0.198 0.107 0.266 0.417 0.443 0.135 0.267 0.045
Table 6: performance across frequency bins. Head/Torso/Tail thresholds: 500/50–500/< training
> 50 examples. = Head F - Tail F (lower is better). Best results in bold.
∗∆
1 1 treme imbalance), vocabulary-constrained LLMs for discovery, knowledge graph construction, and demonstrate superior robustness across frequency temporal analysis at scale. AstroConcepts, by bins, achieving 3× better head-tail balance than tra- contrast, provides controlled-vocabulary annotaditional supervised approaches. For RQ2 (domain tions grounded in the UAT, enabling reproducible adaptation benefits), we find asymmetric improve- evaluation of classification methods under extreme ments that focus on rare, specialized terminology label imbalance, a use case for which a fixed label rather than on frequent concepts, challenging as- space, train/test split, and expert-assigned labels sumptions about uniform domain adaptation effects. are essential. Looking forward, the two resources
RQ3 (competitive LLM performance) is answered open natural avenues for joint investigation: Asaffirmatively: the hybrid approach combining as- troMLab 5 concepts could serve as additional cantroBERT candidate generation with LLM selection didate labels or weak supervision signals for tail achieves competitive results while requiring sub- concepts in AstroConcepts, while UAT-grounded stantially fewer computational resources than full annotations could provide an extrinsic evaluation fine-tuning. signal for the quality of LLM-extracted concept vocabularies.
Methodological Impact We establish frequencystratified evaluation as essential for extreme multilabel assessment and introduce the head-tail gap
(∆) as a robustness metric that reveals perforLimitations and Future Directions Our findings mance patterns invisible in aggregate scores. The are domain-specific and require validation across systematic comparison across paradigms demonother scientific fields before broader claims about strates that different approaches excel in complescientific NLP can be established. The persismentary areas: rule-based methods provide intertent tail performance challenge (best F : 0.198) pretability but limited coverage, k-NN offers high 1 indicates fundamental limitations in current aprecall with domain-adapted embeddings, superproaches, suggesting opportunities for architectural vised models achieve confident predictions, and innovations tailored to extreme imbalance scenarconstrained LLMs balance precision-recall tradeios. Future work should explore integrating strucoffs effectively. tured domain knowledge with few-shot learning approaches. An important validation step is to audit
Broader Implications The torso peak phea stratified sample of the corpus through expert renomenon, in which all methods perform best on view and inter-annotator agreement analysis, which mid-frequency concepts, suggests fundamental limwould quantify annotation noise and its downstream its to learning from extreme imbalance that traneffect on recall-based metrics, particularly for tail scend architectural choices. This finding has imconcepts where incomplete author-assigned labels mediate implications for resource allocation in scimay inflate the apparent difficulty. Extending this entific NLP: optimization efforts should target the methodology to other scientific domains is essential frequency regions where improvement is most feato establish whether the torso-peak phenomenon, sible, rather than aiming for uniform performance asymmetric domain adaptation benefits, and the efgains across all concepts. fectiveness of vocabulary-constrained approaches generalize beyond astrophysics. Finally, more exRelationship to Complementary Resources A tensive experiments with the LLM filtering stage related effort from Ting et al. (2025) extracted 9,999 are needed: evaluating DeepSeek over candidate concepts from 408,590 astrophysics papers using sets generated by BERT and SciBERT, in addition
LLM-based pipelines and clustering over full-text to astroBERT, would isolate the contribution of the content. The two resources address different but candidate generator’s quality from the LLM’s own complementary needs. AstroMLab 5 produces a re-ranking ability, providing a cleaner assessment semantically rich, emergent vocabulary optimized of where the performance gains originate.

<!-- page 9 -->

7. Conclusion 8. Ethical Considerations
All abstracts are from published scientific papers that are publicly accessible. We include only bibliographic metadata (bibcode, title, abstract, publication year, journal) and assigned UAT concepts.
AstroConcepts provides the NLP community
9. Bibliographical References with essential resources for investigating extreme multi-label classification in scientific domains, in particular for astrophysics. Through systematic evaluation across traditional, neural, and
A. Accomazzi, N. Gray, C. Erdmann, C. Biemesvocabulary-constrained approaches, we demonderfer, K. Frey, and J. Soles. 2014. The Unistrate three key insights that advance understandfied Astronomy Thesaurus. In
Astronomical Data ing of scientific text classification. The effectiveXXIII, volume
Analysis Software and Systems ness of hybrid vocabulary-constrained approaches,
485 of
Astronomical Society of the Pacific Conin which astroBERT generates candidate labels
Series, page 461. ference and DeepSeek selects from this constrained set, demonstrates that domain expertise can be incor- Alberto Accomazzi, Michael J. Kurtz, Edwin A. porated through structured vocabulary guidance Henneken, Roman Chyla, James Luker, Carrather than through extensive LLM fine-tuning. This olyn S. Grant, Donna M. Thompson, Alexandra approach achieves competitive performance (F : Holachek, Rahul Dave, and Stephen S. Murray.
1
0.377) while requiring only inference costs for the 2015. Ads: The next generation search platform. domain model and API calls for the LLM, opening
Jennifer Bartlett, Mugdha Polimera, Kelly Lockhart, promising directions for cost-effective scientific NLP
Alberto Accomazzi, Michael Kurtz, and Science that combines specialized knowledge extraction
Explorer Team. 2025. ADS and SciX: Pioneerwith general language understanding capabilities. ing the Next Generation of Interdisciplinary ReDomain adaptation benefits concentrate asymmetsearch Discovery. In rically on rare, specialized terminology, suggest- American Astronomical
#245, volume 245 ing that specialized models primarily handle con- Society Meeting Abstracts of cepts beyond general model capabilities rather than American Astronomical Society Meeting Abstracts, page 442.04. improving performance uniformly. Our frequencystratified evaluation framework, combined with roIz Beltagy, Kyle Lo, and Arman Cohan. 2019. SciBbustness metrics, provides useful methods for asERT: A pretrained language model for scientific sessing extreme multi-label systems in which aggretext. In Proceedings of the 2019 Conference on gate scores can mask critical performance patterns.
Empirical Methods in Natural Language ProcessThese contributions address a critical methodologing and the 9th International Joint Conference on ical gap by enabling systematic evaluation of exNatural Language Processing (EMNLP-IJCNLP), treme imbalance while providing actionable insights pages 3615–3620, Hong Kong, China. Associafor scientific NLP practitioners. The corpus, basetion for Computational Linguistics. lines, and evaluation framework lay the foundations for future research on specialized domain Bishwaranjan Bhattacharjee, Aashka Trivedi, classification, while our findings on vocabulary- Masayasu Muraoka, Muthukumaran Ramasubconstrained approaches indicate promising direc- ramanian, Takuma Udagawa, Iksha Gurung, Nistions for resource-efficient scientific text processing han Pantha, Rong Zhang, Bharath Dandala, systems. To facilitate future research, we make the Rahul Ramachandran, Manil Maskey, Kaylin
AstroConcepts corpus publicly available. The per- Bugbee, Mike Little, Elizabeth Fancher, Irina sistent challenges in tail performance underscore Gerasimov, Armin Mehrabian, Lauren Sanders, opportunities for novel approaches that integrate Sylvain Costes, Sergi Blanco-Cuaresma, Kelly structured knowledge with text-based classification. Lockhart, Thomas Allen, Felix Grezes, Megan
Future work should prioritize annotation validation Ansdell, Alberto Accomazzi, Yousef El-Kurdi, through expert audits and inter-annotator agree- Davis Wertheimer, Birgit Pfitzmann, Cesar ment studies, systematic cross-domain replication, Berrospi Ramis, Michele Dolfi, Rafael Teixeira de and controlled ablations of the LLM re-ranking Lima, Panagiotis Vagenas, S. Karthik Mukkavstage across candidate generators of varying qual- illi, Peter Staar, Sanaz Vahidinia, Ryan Mcity. As the scientific literature continues to expand Granaghan, and Tsendgar Lee. 2024. INDUS: and specialized terminology increases, effective Effective and Efficient Language Models for handling of extreme imbalance becomes essential Scientific Applications. e-prints, page arXiv for scientific NLP applications. arXiv:2405.10725.

<!-- page 10 -->

Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Hao, Zhibin Gou, Zhicheng Ma, Zhigang Yan,
Downey, and Daniel Weld. 2020. SPECTER: Zhihong Shao, Zhipeng Xu, Zhiyu Wu, Zhongyu
Document-level representation learning using Zhang, Zhuoshu Li, Zihui Gu, Zijia Zhu, Zijun citation-informed transformers. In Liu, Zilin Li, Ziwei Xie, Ziyang Song, Ziyi Gao,
Proceedings and Zizheng Pan. 2024. DeepSeek-V3 Technical of the 58th Annual Meeting of the Association
Linguistics, pages 2270–2282, Report. e-prints, page arXiv:2412.19437. for Computational arXiv
Online. Association for Computational LinguisJacob Devlin, Ming-Wei Chang, Kenton Lee, and tics.
Kristina Toutanova. 2019. BERT: Pre-training
DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingx- of deep bidirectional transformers for language uan Wang, Bochao Wu, Chengda Lu, Cheng- understanding. In
Proceedings of the 2019 Congang Zhao, Chengqi Deng, Chenyu Zhang, ference of the North American Chapter of the AsChong Ruan, Damai Dai, Daya Guo, Dejian Yang, sociation for Computational Linguistics: Human
Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin,
Language Technologies, Volume 1 (Long and
Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Papers), pages 4171–4186, Minneapolis,
Short
Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Minnesota. Association for Computational LinXu, Haocheng Wang, Haowei Zhang, Honghui guistics.
Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu,
C. Lee Giles, Kurt D. Bollacker, and Steve
J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, JiLawrence. 1998. Citeseer: an automatic citaashi Li, Jiawei Wang, Jin Chen, Jingchang Chen, tion indexing system. In
Jingyang Yuan, Junjie Qiu, Junlong Li, Junxiao Proceedings of the Third
Libraries, DL ’98,
Song, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, ACM Conference on Digital page 89–98, New York, NY, USA. Association
Kexin Huang, Kuai Yu, Lean Wang, Lecong for Computing Machinery.
Zhang, Lei Xu, Leyi Xia, Liang Zhao, Litong
Wang, Liyue Zhang, Meng Li, Miaojun Wang,
F. Grezes, S. Blanco-Cuaresma, A. Accomazzi,
Mingchuan Zhang, Minghua Zhang, Minghui
M. J. Kurtz, G. Shapurian, E. Henneken, C. S.
Tang, Mingming Li, Ning Tian, Panpan Huang,
Grant, D. M. Thompson, R. Chyla, S. McDonald,
Peiyi Wang, Peng Zhang, Qiancheng Wang, QiT. W. Hostetler, M. R. Templeton, K. E. Lockhart, hao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen,
N. Martinovic, S. Chen, C. Tanner, and P. ProR. L. Jin, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, topapas. 2024. Building astroBERT, a Language
Runji Wang, Runxin Xu, Ruoyu Zhang, Ruyi
Model for Astronomy & Astrophysics. In
Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, AstromiXXXI,
Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, cal Data Analysis Software and Systems volume 535 of
Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Astronomical Society of the Pacific
Series, page 119.
Zhou, Shuiping Yu, Shunfeng Zhou, Shuting Pan, Conference
T. Wang, Tao Yun, Tian Pei, Tianyu Sun, W. L.
Yi Huang, Buse Giledereli, Abdullatif Köksal, ArzuXiao, Wangding Zeng, Wanjia Zhao, Wei An, can Özgür, and Elif Ozkirimli. 2021. Balancing
Wen Liu, Wenfeng Liang, Wenjun Gao, Wenmethods for multi-label text classification with qin Yu, Wentao Zhang, X. Q. Li, Xiangyue Jin, long-tailed class distribution. In
Proceedings
Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan of the 2021 Conference on Empirical Methods
Wang, Xiaojin Shen, Xiaokang Chen, Xiaokang
Processing, pages 8153– in Natural Language
Zhang, Xiaosha Chen, Xiaotao Nie, Xiaowen
8161, Online and Punta Cana, Dominican ReSun, Xiaoxiang Wang, Xin Cheng, Xin Liu, Xin public. Association for Computational Linguistics.
Xie, Xingchao Liu, Xingkai Yu, Xinnan Song,
Xinxia Shan, Xinyi Zhou, Xinyu Yang, Xinyuan Li,
Kamran Kowsari, Donald E. Brown, Mojtaba HeiXuecheng Su, Xuheng Lin, Y. K. Li, Y. Q. Wang, darysafa, K. Meimandi, Matthew S. Gerber, and
Y. X. Wei, Y. X. Zhu, Yang Zhang, Yanhong Xu,
Laura E. Barnes. 2017. Hdltex: Hierarchical deep
Yanhong Xu, Yanping Huang, Yao Li, Yao Zhao, learning for text classification. 2017 16th IEEE
Yaofeng Sun, Yaohui Li, Yaohui Wang, Yi Yu,
International Conference on Machine Learning
Yi Zheng, Yichao Zhang, Yifan Shi, Yiliang Xiong, and Applications (ICMLA), pages 364–371.
Ying He, Ying Tang, Yishi Piao, Yisong Wang,
Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang
David Lewis. 1987. Reuters-21578 Text CategorizaGuo, Yu Wu, Yuan Ou, Yuchen Zhu, Yuduan tion Collection. UCI Machine Learning ReposiWang, Yue Gong, Yuheng Zou, Yujia He, Yukun tory. DOI: https://doi.org/10.24432/C52G6M.
Zha, Yunfan Xiong, Yunxian Ma, Yuting Yan,
Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Rundong Liu, Wenhan Liang, Weijun Luo, Yuxiang
Zhou, Z. F. Wu, Z. Z. Ren, Zehui Ren, Zhangli Song, He Zhang, Ruohua Xu, Yunfeng Li, and
Sha, Zhe Fu, Zhean Xu, Zhen Huang, Zhen Ming Liu. 2023. Recent advances in hierarchical
Zhang, Zhenda Xie, Zhengyan Zhang, Zhewen multi-label text classification: A survey.

<!-- page 11 -->

Eneldo Loza Mencía and Johannes Fürnkranz. Pengcheng Yang, Xu Sun, Wei Li, Shuming Ma,
2010. Wei Wu, and Houfeng Wang. 2018. SGM: SeEfficient Multilabel Classification Algoquence generation model for multi-label classifirithms for Large-Scale Problems in the Legal
Domain, pages 192–215. Springer Berlin Hei- cation. In
Proceedings of the 27th International delberg, Berlin, Heidelberg. Linguistics, pages
Conference on Computational
3915–3926, Santa Fe, New Mexico, USA. AssoAndrew McCallum, Kamal Nigam, Jason D. M. Renciation for Computational Linguistics. nie, and Kristie Seymore. 2000. Automating the construction of internet portals with machine Min-Ling Zhang and Zhi-Hua Zhou. 2014. A review learning. Retrieval, 3:127–163. on multi-label learning algorithms. TKDE.
Information IEEE
Alistair Miles and Sean Bechhofer. 2009. SKOS Chuang Zhou, Junnan Dong, Xiao Huang, Zirui
Simple Knowledge Organization System Refer- Liu, Kaixiong Zhou, and Zhaozhuo Xu. 2024. ence. W3c recommendation, W3C. QUEST: Efficient extreme multi-label text classification with large language models on commodMobashir Sadat and Cornelia Caragea. 2022. Hierity hardware. In
Findings of the Association for archical multi-label classification of scientific doc2024, pages
Computational Linguistics: EMNLP uments. In
Proceedings of the 2022 Conference
3929–3940, Miami, Florida, USA. Association for on Empirical Methods in Natural Language ProComputational Linguistics. cessing, pages 8923–8937, Abu Dhabi, United
Arab Emirates. Association for Computational
Linguistics.
A.1. Complete Prompt Template
António Paulo Santos and Fátima Rodrigues. 2009. (Section 4.2.3)
Multi-label hierarchical text classification using the acm taxonomy. Figure 2 shows the designed prompt as part of our experiments.
Seyed Amin Tabatabaei, Sarah Fancher, Michael
Parsons, and Arian Askari. 2025. Can large lanA.2. astroBERT label coverage guage models serve as effective classifiers for
(Section 4.2.3) hierarchical multi-label classification of scientific documents at industrial scale? In
Proceedings
Figure 3 shows that astroBERT’s top-50 predicted of the 31st International Conference on Compulabels covers approximately 82% of the groundTrack, pages 163– tational Linguistics: Industry truth labels.
174, Abu Dhabi, UAE. Association for Computational Linguistics.
Yuan-Sen Ting, Alberto Accomazzi, Tirthankar
Ghosal, Tuan Dung Nguyen, Rui Pan, Zechang
Sun, and Tijmen de Haan. 2025. AstroMLab 5:
Structured summaries and concept extraction for
400,000 astrophysics papers. In Proceedings of the Third Workshop for Artificial Intelligence for
Scientific Publications, pages 170–185, Mumbai,
India and virtual. Association for Computational
Linguistics.
Autumn Toney and James Dunham. 2022. Multilabel classification of scientific research documents across domains and languages. In
Proceedings of the Third Workshop on ScholProcessing, pages 105–114, arly Document
Gyeongju, Republic of Korea. Association for
Computational Linguistics.
Grigorios Tsoumakas and Ioannis Katakis. 2010.
Mining multi-label data. Data Mining and Knowledge Discovery Handbook.
Ximing Wu, Hao Chen, and Qinmin Zhang. 2020.
Revisiting macro-f1 score for imbalanced multilabel classification. preprint. arXiv

<!-- page 12 -->

You are an expert astrophysicist and scientific topic classifier.
Your task is to choose between 1 to 10 labels from the candidate list that accurately describe the main scientific themes of the following research paper.
A label should be selected only if it is clearly relevant to the paper’s content.
–Abstract:
{abstract}
–Candidate topics suggested by the model:
{topk_labels}
Return your answer in valid JSON as follows:
{
"selected_labels": ["label1", "label2", ...]
}
Do not include explanations or text outside the JSON.
Figure 2: Prompt template used for vocabulary-constrained LLM classification.
Figure 3: Fine-tuned astroBERT Label Coverage
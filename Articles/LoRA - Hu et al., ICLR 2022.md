Published as a conference paper at ICLR 2025 

## COMPUTATIONAL LIMITS OF LOW-RANK ADAPTATION (LORA) FINE-TUNING FOR TRANSFORMER MODELS<sup>_∗_</sup> 

**Jerry Yao-Chieh Hu**<sup>_†_</sup> **Maojiang Su**<sup>_‡_</sup> **En-Jui Kuo**<sup>_♭_</sup> **Zhao Song**<sup>_§_</sup> **Han Liu**<sup>_†_</sup> 

> _†_ Northwestern University _‡_ University of Science and Technology of China 

> _♭_ National Yang Ming Chiao Tung University _§_ Simons Institute, UC Berkeley 

jhu@u.northwestern.edu, sumaojiang@mail.ustc.edu.cn, kuoenjui@nycu.edu.tw, magic.linuxkde@gmail.com, hanliu@northwestern.edu 

### ABSTRACT 

We study the computational limits of Low-Rank Adaptation (LoRA) for finetuning transformer-based models using fine-grained complexity theory. Our key observation is that the existence of low-rank decompositions within the gradient computation of LoRA adaptation leads to possible algorithmic speedup. This allows us to (i) identify a phase transition behavior of efficiency assuming the Strong Exponential Time Hypothesis (SETH), and (ii) prove the existence of almost linear algorithms by controlling the LoRA update computation term by term. For the former, we identify a sharp transition in the efficiency of all possible rank- _r_ LoRA update algorithms for transformers, based on specific norms resulting from the multiplications of the input sequence _X_ , pretrained weights _W_<sup>_⋆_</sup> , and adapter matrices _αBA/r_ . Specifically, we derive a shared upper bound threshold for such norms, and show that efficient (sub-quadratic) approximation algorithms of LoRA exist only below this threshold. For the latter, we prove the existence of almost linear approximation algorithms for LoRA adaptation by utilizing the hierarchical low-rank structures of LoRA gradients and approximating the gradients with a series of chained low-rank approximations. To showcase our theory, we consider two practical scenarios: partial (e.g., only _WV_ and _WQ_ ) and full adaptations (e.g., _WQ_ , _WV_ , and _WK_ ) of weights in attention heads. 

### 1 INTRODUCTION 

We investigate the computational limits of finetuning large transformer-based pretrained model with **Lo** w- **R** ank **A** daptation ( **LoRA** ). This analysis is of practical importance in the era of Large Foundation Models (Bommasani et al., 2021). Large foundation models are gigantic transformerbased architectures, pretrained on vast datasets, are pivotal across multiple fields, including natural language processing (Achiam et al., 2023; Touvron et al., 2023b;a; Brown et al., 2020; Floridi and Chiriatti, 2020), finance (Yang et al., 2023; Wu et al., 2023), genomics (Nguyen et al., 2024; Zhou et al., 2025; 2024; 2023; Ji et al., 2021), medical science (Thirunavukarasu et al., 2023; Singhal et al., 2023; Moor et al., 2023) and more. They are powerful but very expensive to pretrain. Therefore, most practitioners rely on finetuing methods to adapt these models for their specific needs (Zheng et al., 2024; Ding et al., 2022). LoRA (Mao et al., 2025; Hu et al., 2021) is the most prevalent fine-tuning method due to its parameter efficiency due to the low-rank adaptation of model weights. However, even with LoRA, updating the partial weights of pretrained transformer-based models using gradient methods remains costly. Notably, the naive backward pass in transformer architectures retains the same quadratic-in-sequence-length computational time complexity as its forward pass (see Appendix F for discussions and a proof). This work provides a timely theoretical analysis of LoRA’s computational limits, aiming to advance efficient finetuning of large foundation models. 

The hardness of LoRA finetuning transformer-based foundation model ties to both forward and backward passes. To analyze, it suffices to focus on just transformer attention heads due to their dominating quadratic time complexity in both passes. We first make the following observation: 

> _∗_ Code is available on OpenReview; full version and future updates are on arXiv. 

1 

Published as a conference paper at ICLR 2025 

The hardness of LoRA’s forward pass is trivially characterized by (Alman and Song, 2023). 

To see this, let _X ∈_ R<sup>_L×d_</sup> be input with length _L_ , and _WK, WQ, WV ∈_ R<sup>_d×d_</sup> be attention weights, and _Q_ = _XWV ∈_ R<sup>_L×d_</sup> , _K_ = _XWK ∈_ R<sup>_L×d_</sup> , _V_ = _XV ∈_ R<sup>_L×d_</sup> . The Attention Mechanism is 

_Z_ = Softmax � _QK_<sup>T</sup> _β_ � _V_ = _D_<sup>_−_1</sup> exp� _XWQWK_<sup>T</sup><sup>_X_T</sup><sup>_β_</sup> � _XWV ,_ (1.1) 

with the inverse temperature _β >_ 0 and _D_ := diag �exp� _XWQWK_<sup>T</sup><sup>_X_T</sup><sup>_β_</sup> �1 _L_ �. Here, exp( _·_ ) is _·_ entry-wise exponential function, diag ( ) converts a vector into a diagonal matrix with the entries of the vector, and 1 _L_ is the length- _L_ all ones vector. LoRA finetuning is given as 

**Definition 1.1** (LoRA (Hu et al., 2021)) **.** Let _W ∈_ R<sup>_b×a_</sup> be any weight matrix in a pretrained model _F_ , LoRA fine-tunes _F_ through updating _W_ with a low-rank decomposition _W_ = _W_<sup>_⋆_</sup> +<sup>_<u>α</u>_</sup> _r_<sup>_BA_.Here,</sup> _W_<sup>_⋆_</sup> is the frozen pretrained weight. Only _B ∈_ R<sup>_b×r_</sup> and _A ∈_ R<sup>_r×a_</sup> are learnable (being update via gradient descent) with rank _r <_ min( _a, b_ ) and tunable hyperparameter _α ∈_ R. 

Under the Strong Exponential Time Hypothesis (Hypothesis 1), Alman and Song (2023) state: 

**Lemma 1.1** (Informal, (Alman and Song, 2023)) **.** Fast (sub-quadratic) forward pass of transformer only exist when entries of _K, Q, V_ are bounded by a constant _B_ = Θ(<sup>_√_</sup> log _L_ ). 

It is easy to see that Lemma 1.1 is transferable to LoRA inference according to Definition 1.1. However, we still need the hardness of backward pass to fully characterize LoRA for transformers. The analysis of the backpropagation (backward pass) is less straightforward. It involves managing the computation of numerous gradients for attention scores, with the number of chain-rule terms scaling quadratically in _L_ and the numbers of LoRA weights. While it is tempting to design algorithms to circumvent this Ω( _L_<sup>2</sup> ) computation time, to the best of our knowledge, there are no formal results to support and characterize such algorithms. To address this gap, we pose the following questions and provide a fundamental theory to fully characterize the complexity of LoRA for transformer models: 

**Question 1.** Is it possible to improve the Ω( _L_<sup>2</sup> ) time with a bounded approximation error? 

**Question 2.** More aggressively, is it possible to do such gradient computations in almost linear time? 

To address these questions, we explore approximate LoRA gradient computations with precision guarantees. We first layout the objective of finetuning transformer-based pretrained models. 

**Definition 1.2** (LoRA Loss for Adapting _WK_ , _WQ_ , _WV_ of an Attention Head) **.** Let _D_ = _{Xi, Yi}_<sup>_N_</sup> _i_ =1 be a dataset of size _N_ with _Xi ∈_ R<sup>_L×d_</sup> being the input and _Yi ∈_ R<sup>_L×d_</sup> being the label. Fine-tuning a (self-)attention with LoRA with _ℓ_ 2 loss on dataset _D_ is formulated as 



We study the following approximation problem. Let _<u>Z</u>_ := vec( _Z_ ) _∈_ R<sup>_ab_</sup> for any matrix _Z ∈_ R<sup>_a×b_</sup> . 

**Problem 1** (Approximate LoRA Gradient Computation (ALoRAGC( _L, d, r, ϵ_ ))) **.** Assume all numerical values in log( _L_ ) bits encoding. Let _L_ follow Definition 1.2. The problem of approximating gradient computation of optimizing (1.2) is to find six surrogate gradient matrices _{G_<sup>�(</sup> _µ_<sup>_A_)</sup> _∈_ R<sup>_d×r_</sup> _, G_<sup>�(</sup> _µ_<sup>_B_)</sup> _∈_ R<sup>_r×d_</sup> _}µ_ = _K,Q,V_ such that max ������ _G_ ( _µB_ ) _− ∂B∂Lµ_ ��� _∞_<sup>_,_</sup> ���� _G_ ( _µA_ ) _− ∂A∂Lµ_ ��� _∞_ � _µ_ = _K,Q,V_ � _≤ ϵ,_ for some _ϵ >_ 0, where _∥Z∥∞_ := max _i,j |Zij|_ . 

**Remark 1.1.** Any method or algorithm that aims to compute LoRA gradients beyond vanilla computation of (1.2) falls within the scope of this problem. Examples include using sampling strategies to avoid full LoRA gradient computation (Pan et al., 2024) or employing model quantization 

2 

Published as a conference paper at ICLR 2025 

for efficiency via low-precision gradient computation (Li et al., 2024; Dettmers et al., 2024). Common among these approaches is the need to compute surrogate LoRA gradients with reduced computational cost. We abstract this key subroutine and consider the fundamental algorithmic Problem 1. 

In this work, we aim to investigate the computational limits of all possible efficient algorithms of ALoRAGC( _L, d, r, ϵ_ ) under realistic setting _ϵ_ = 1 _/_ poly( _L_ ). 

**Contributions.** Our contributions are 2-fold: 

- **Norm-Based Phase Transition of Efficiency (Theorem 4.1).** We answer Question 1 by identifying a phase transition behavior on the norm of input, pretrained and adaptor weights, assuming the Strong Exponential Time Hypothesis (SETH). Specifically, we identify an inefficiency threshold for these norms such that, only below which, adapting transformer-based models with LoRA in _L_<sup>2</sup><sup>_−o_(1)</sup> (sub-quadratic) time is possible. 

**Theorem 1.1** (Informal Version of Theorem 4.1) **.** Without appropriately normalized inputs _X_ , pretrained attention weights _WK_<sup>_⋆, W_</sup> _Q_<sup>_⋆, W_</sup> _V_<sup>_⋆_, and LoRA matrices</sup><sup>_{αAµBµ/r}µ_=</sup><sup>_K,Q,V_, there is no</sup> algorithm running in subquadratic time _O_ ( _L_<sup>2</sup><sup>_−δ_</sup> ) for any constant _δ >_ 0 to solve ALoRAGC. 

- **Existence of Almost Linear Time LoRA Algorithms.** We answer Question 2 by proving that precision-guaranteed approximation to Problem 1 is achievable in _almost linear time_ via hierarchical low-rank decomposition of LoRA gradients. To showcase our theory, we analyze two practical scenarios highlighted in (Hu et al., 2021): _partial_ adaptations (e.g., only _WV_ and _WQ_ in Section 3), and _full_ adaptations (e.g., _WK, WQ, WV_ in Appendix A) of weights in attention heads. 

**Theorem 1.2** (Informal Version of Theorems 3.1 and A.1) **.** Given appropriately normalized inputs _X_ , pretrained attention weights _WK_<sup>_⋆, W_</sup> _Q_<sup>_⋆, W_</sup> _V_<sup>_⋆_, and LoRA matrices</sup><sup>_{αAµBµ/r}µ_=</sup><sup>_K,Q,V_, there</sup> exists an algorithm that solves ALoRAGC in almost linear time _O_ ( _L_<sup>1+</sup><sup>_o_(1)</sup> ). 

On the theoretical front, we characterize the computational feasibility of LoRA by showing the existence of precision-guaranteed, efficient (subquadratic or almost linear time) LoRA methods and identifying their necessary conditions. On the practical front, these conditions serve as valuable guidelines for implementations (please see Remark 5.2 for discussions and Appendix G for numerical justifications). Importantly, our theory only requires one assumption on numerical value encoding (e.g., in log _L_ bits with _L_ being the sequence length). Such an assumption is minimal and realistic. No assumptions are made about the data or model, making our results widely applicable. 

**Organization.** Section 2 includes preliminaries and problem setup. Section 3 presents analysis of LoRA adaptation on only _WQ, WK_ . Appendix A presents analysis of LoRA adaptation on all _WQ, WK, WV_ . Section 4 characterizes the computational limits of all possible efficient algorithms for LoRA. Section 5 includes concluding remarks. We defer discussions of related works to Appendix B. 

**Notations.** We denote (column) vectors by lower case letters, and matrices by upper case letters. Let 1 _L_ denote the length- _L_ all ones vector. We write _⟨a, b⟩_ := _a_<sup>T</sup> _b_ as the inner product for vectors _a, b_ . Let _a_ [ _i_ ] denotes the _i_ -th component of vector _a_ . Let _A_ [ _i, j_ ] and _Aij_ denotes the ( _i, j_ )-th entry of matrix _A_ . For any matrix _A_ , let _A_ [ _i, ·_ ] and _A_ [ _·, j_ ] be the _i_ -th row and _j_ -th column of _A_ , respectively. For _u, v ∈_ R<sup>_d_</sup> , we denote their Hadamard product as _u ⊙ v_ := ( _u_ 1 _v_ 1 _, . . . , udvd_ )<sup>T</sup> . The index set _{_ 1 _, · · · , I}_ is denoted by [ _I_ ], where _I ∈_ N+. For any _z ∈_ R<sup>_d_</sup> , we denote exp( _z_ ) _∈_ R<sup>_d_</sup> whose _i_ -th entry is exp( _zi_ ). Let _∥A∥∞_ := max _i,j |Aij|_ for any matrix _A_ . Let _∥·∥F_ denote the squared Frobenius norm, i.e., _∥A∥F_ := (<sup>�</sup> _i,j_<sup>_A_2</sup> _ij_<sup>)1</sup><sup>_/_2.</sup> 

### 2 PRELIMINARIES AND PROBLEM SETUP 

This section presents the ideas we build on. 

**Tensor Trick for Computing Gradients.** The tensor trick (Diao et al., 2019; 2018) is an instrument to compute complicated gradients in a clean and tractable fashion. As we shall see below, the purpose of the tensor trick is to convert matrix multiplication into vector form, making the gradient w.r.t. the matrix more tractable. For this, we introduce vectorization and its inverse operation, matrixization. 

3 

Published as a conference paper at ICLR 2025 

**Definition 2.1** (Vectorization) **.** For any matrix _X ∈_ R<sup>_L×d_</sup> , we define _<u>X</u>_ := vec ( _X_ ) _∈_ R<sup>_Ld_</sup> such that _Xi,j_ = _<u>X</u>_ ~~(~~ _i−_ 1) _d_ + _j_<sup>for all</sup><sup>_i ∈_[</sup><sup>_L_] and</sup><sup>_j∈_[</sup><sup>_d_].</sup> 

**Definition 2.2** (Matrixization) **.** For any vector _<u>X</u> ∈_ R<sup>_Ld_</sup> , we define mat( _<u>X</u>_ <u>)</u> = _X_ such that _Xi,j_ = mat( _<u>X</u>_ ) := _<u>X</u>_ ~~(~~ _i−_ 1) _d_ + _j_<sup>for all</sup><sup>_i ∈_[</sup><sup>_L_] and</sup><sup>_j∈_[</sup><sup>_d_], namely mat(</sup><sup>_·_) = vec</sup><sup>_−_1(</sup><sup>_·_).</sup> 

Next, we introduce necessary tensor terminologies. 

**Definition 2.3** (Kronecker Product) **.** Let _A ∈_ R<sup>_La×da_</sup> and _B ∈_ R<sup>_Lb×db_</sup> . We define the Kronecker product of _A_ and _B_ as _A ⊗ B ∈_ R<sup>_LaLb×dadb_</sup> such that ( _A ⊗ B_ )( _ia−_ 1) _Lb_ + _ib,_ ( _ja−_ 1) _db_ + _jb_ , is equal to _Aia,ja Bib,jb_ with _ia ∈_ [ _La_ ] _, ja ∈_ [ _da_ ] _, ib ∈_ [ _Lb_ ] _, jb ∈_ [ _db_ ]. 

**Definition 2.4** (Sub-Block of a Tensor) **.** For any _A ∈_ R<sup>_La×da_</sup> and _B ∈_ R<sup>_Lb×db_</sup> , let A := _A ⊗ B ∈_ R<sup>_LaLb×dadb_</sup> . For any _<u>j</u> ∈_ [ _La_ ], we define A _<u>j</u> ∈_ R<sup>_Lb×dadb_</sup> be the _<u>j</u>_ -th _Lb × dadb_ sub-block of A. 

Definition 2.3 creates a large matrix from two smaller matrices, preserving the structure and properties of the original matrices. Definition 2.4 provides a refined identification of specific entry-wise multiplications between the two _Kronecker-producted_ matrices. Together, they makes the gradient w.r.t. the matrix more tractable: for instance, the gradient of below vectorized LoRA loss (2.1). 

**Lemma 2.1** (Tensor Trick (Diao et al., 2019; 2018)) **.** For any _A ∈_ R<sup>_La×da_</sup> , _B ∈_ R<sup>_Lb×db_</sup> and _X ∈_ R<sup>_da×db_</sup> , it holds vec � _AXB_<sup>T�</sup> = ( _A ⊗ B_ <u>)</u> _<u>X</u> ∈_ R<sup>_LaLb_</sup> . 

To showcase the tensor trick for LoRA, let’s consider a (single data point) simplified (1.2) 



By Definition 2.3 and Definition 2.4, we identify _Dj,j_ := exp A _<u>j W</u> ,_ 1 _L ∈_ R for all _<u>j</u> ∈_ [ _L_ ], � <u>� � �</u> with A := _X ⊗ X ∈_ R<sup>_L_2</sup><sup>_×d_2</sup> and _<u>W</u> ∈_ R<sup>_d_2</sup> . Therefore, for each _<u>j</u> ∈_ [ _L_ ] and _<u>i</u> ∈_ [ _d_ ], it holds 



Gao et al. (2023a;b) show that (2.1) provides term-by-term tractability for gradient computation of _L_ 0. Specifically, it allow us to convert the attention score _D_<sup>_−_1</sup> exp� _XWX_<sup>T�</sup> into its vectorized form ( _D ⊗ IL_ )<sup>_−_1</sup> exp(A _<u>W</u>_ <u>)</u> _∈_ R<sup>_L_2</sup> and split the vectorized form into _L_ terms of size _L_ . This provides a systematic way to manage the chain-rule terms in the gradient computation of losses like _L_ 0, and opens the door to more general analytical feasibility for deep transformer-based models. 

**Problem Setup: Which Attention Weights in Transformer Should We Apply LoRA to?** Following (Hu et al., 2021), we consider only adapting the attention weights for downstream tasks. This consideration is sufficient to justify our techniques as the attention head dominates the time complexity of transformer-based foundation models. Namely, we consider updating (as in Definition 1.2) 



Furthermore, for completeness, we consider two de facto scenarios as in (Hu et al., 2021, Sec. 7.1): 

(C1) **Special Case.** Adapting only _WQ_ and _WV_ for best performance under fixed parameter budge. 

(C2) **General Case.** Adapting _WK, WQ, WV_ for best performance. 

We analyze (C1) **Special Case** in Section 3 and (C2) **General Case** in Appendix A. 

To consider the problem of adapting attention head, we first generalize Definition 1.2 to the following generic attention with triplet input sequences. For reasons, this allows our results to be applicable. Moreover, this helps us to focus on parts dominating the efficiency of gradient computation. 

4 



<!-- Start of picture text -->
><br>-r ] { C)} |<br>Ci (df)<br>- )<br>o>) en OO<br>( { ( )})<br><!-- End of picture text -->



<!-- Start of picture text -->
6 0) ()<br>( )<br><!-- End of picture text -->

Published as a conference paper at ICLR 2025 

Thus, we deduce Definition 3.1 to 





We introduce the next problem to characterize all possible (efficient or not) gradient computation of optimizing (3.3). Let _Y_ [ _i, ·_ ] and _Y_ [ _·, j_ ] be the _i_ -th row and _j_ -th column of _Y_ , respectively. 

**Problem 2** (Approximate LoRA Gradient Computation ALoRAGC( _L, d, r, ϵ_ )) **.** Given _Ci_<sup>(1)</sup> _, Ci_<sup>(2)</sup> _, Ci_<sup>(3)</sup> _, Yi ∈_ R<sup>_L×d_</sup> . Let _ϵ >_ 0. Assume all numerical values are in log( _L_ )-bits encoding. Let _L_ follows (3.3). The problem of approximating gradient computation of optimizing (3.3) is to find two matrices _G_<sup>�(</sup> _Q_<sup>_A_)</sup> _∈_ R<sup>_d×r_</sup> and _G_<sup>�(</sup> _Q_<sup>_B_)</sup> _∈_ R<sup>_r×d_</sup> such that ( _B_ ) ( _A_ ) max � _∥G_<sup>�</sup> _~~Q~~ − ∂_<sup>_∂_</sup> _<u>B</u>_<sup>_L_</sup> _∥∞,_ _<u>∥G</u>_<sup>�</sup> _~~Q~~ − ∂_<sup>_∂_</sup> _<u>A</u>_<sup>_L_</sup> _∥∞_ � _≤ ϵ._ _~~Q Q~~_ 

The explicit gradient of LoRA loss (3.3) is too complicated to characterize Problem 2. To combat this, we employ the tensor trick. Let _W_ := _W_<sup>_⋆_</sup> _Q_<sup>+</sup><sup>_BQAQ∈_R</sup><sup>_d×d_such that vec (</sup><sup>_W_) =</sup><sup>_<u>W</u>_</sup> _∈_ R<sup>_d_2</sup> . 



Definition 3.2 decomposes the complicated matrix exp� _C_<sup>(1)</sup> ( _W_<sup>_⋆_</sup> _Q_<sup>+</sup><sup>_BQAQ_)(</sup><sup>_C_</sup> _i_<sup>(2)</sup> )<sup>T�</sup> in loss (3.3) into _L_ vectors. Importantly, since the weight _W_ is vectorized into _<u>W</u>_ <u>, such a vectorized representation</u> allows more tractable gradient computation by its term-by-term identifiability. **Definition 3.3** (Attention Score Normalization) **.** Let C := _C_<sup>(1)</sup> _⊗ C_<sup>(2)</sup> such that C _<u>j</u> ∈_ R<sup>_L×d_2</sup> for all _<u>j</u> ∈_ [ _L_ ]. For every _<u>j</u> ∈_ [ _L_ ], we define _α_ ( _x_ <u>)</u> _<u>j</u>_ : R<sup>_d_2</sup> _→_ R as: _α_ <u>(</u> _<u>W</u>_ <u>)</u> _<u>j</u>_ := exp C _<u>jW</u> ,_ 1 _L ∈_ R. � � <u>�</u> � 

Similarly, Definitions 3.2 and 3.3 provide analytical tractability of the matrix _D_ in loss (3.3). 

**Definition 3.4** (Vectorized, Normalized Attention Score) **.** For a fixed _<u>j</u> ∈_ [ _L_ ], we define _f_ <u>(</u> _<u>W</u>_ <u>)</u> _<u>j</u>_ : R<sup>_d_2</sup> _→_ R<sup>_L_</sup> as: _f_ <u>(</u> _<u>W</u>_ <u>)</u> _<u>j</u>_ := _α_ <u>(</u> _<u>W</u>_ <u>)</u><sup>_−_</sup> _<u>j</u>_<sup>1</sup> _u_ <u>(</u> _<u>W</u>_ <u>)</u> _<u>j</u>_ such that _f_ <u>(</u> _<u>W</u>_ <u>)</u> _∈_ R<sup>_L×L_</sup> denotes the matrix whose _<u>j</u>_ -th row is ( _f_ <u>(</u> _<u>W</u>_ <u>)</u> _<u>j</u>_ <u>)</u><sup>_⊤_</sup> . 

Definition 3.4 decomposes the complicated matrix multiplication _D_<sup>_−_1</sup> exp� _C_<sup>(1)</sup> ( _WQ_<sup>_⋆_+</sup><sup>_BQAQ_)(</sup><sup>_C_(2))T�</sup> _C_<sup>(3)</sup> in loss (3.3) into _L_ terms. Note that the gradients w.r.t. _<u>W</u>_ are still tractable due to simple chain rule (by design of _α_ ( _·_ ) and _u_ ( _·_ )). 

**Definition 3.5** (Vectorized LoRA Loss (3.3)) **.** For every _i ∈_ [ _d_ ], let _C_<sup>(3)</sup> [ _·, i_ ] follow (S2). For every _<u>j</u> ∈_ [ _L_ ] and _i ∈_ [ _d_ ], we define _c_ ( _x_ <u>)</u> _<u>j,i</u>_<sup>:R</sup><sup>_d_2</sup><sup>_×_R</sup><sup>_d_2</sup><sup>_→_R as:</sup><sup>_c_</sup><sup><u>(</u></sup><sup>_<u>W</u>_</sup> <u>)</u> _<u>j,i</u>_<sup>:=</sup><sup>_⟨f_</sup><sup><u>(</u></sup><sup>_<u>W</u>_</sup> <u>)</u> _<u>j, C</u>_<sup>(3)</sup> [ _·, i_ ] _⟩− Yj,i_<sup>.</sup> Here _Yj,i_<sup>=</sup><sup>_Y_</sup><sup><u>[</u></sup><sup>_<u>j</u>_</sup> _<u>, i</u>_ ] is the <u>(</u> _<u>j, i</u>_ )-th entry of _Y ∈_ R<sup>_L×d_</sup> for _<u>j</u> ∈_ [ _L_ ] _, i ∈_ [ _d_ ]. 

From above definitions, we read out _c_ <u>(</u> _<u>W</u>_ <u>) =</u> _f_ <u>(</u> _<u>W</u>_ <u>)</u> _C_<sup>(3)</sup> _− Y_ such that (3.3) becomes 



(3.4) presents a decomposition of the LoRA loss (3.3) into _L · d_ terms, each simple enough for tracking gradient computation. Now, we are ready to compute the gradient of the LoRA loss. 

6 

Published as a conference paper at ICLR 2025 

**Lemma 3.1** (Low-Rank Decomposition of LoRA Gradient) **.** Let matrix _BQ, AQ_ and loss function _L_ follow (3.3), _W_ := _W_<sup>_⋆_</sup> _Q_<sup>+</sup><sup>_BQAQ_and C :=</sup><sup>_C_(1)</sup><sup>_⊗C_(2).It holds</sup> 



#### _Proof._ See Appendix C.1 for a detailed proof. 

**Remark 3.1** (Benefit from Tensor Trick: Fast Approximation) **.** As we shall show in subsequent sections, Lemma 3.1 also enables the construction of fast approximation algorithms for (3.5) with precision guarantees due to its analytical feasibility. Surprisingly, it is even possible to compute (3.5) in almost linear time. To proceed, we further decompose (3.5) into its fundamental building blocks according to the chain-rule in the next lemma, and then conduct the approximation term-by-term. 

**Remark 3.2** (LoRA Gradient Computation Takes Quadratic Time) **.** Lemma 3.1 implies that LoRA’s gradient computation takes quadratic time, similar to inference hardness result (Alman and Song, 2023). This is non-trivial yet not the main focus of this work. Please see Appendix F for details. 



_Proof._ See Appendix C.2 for a detailed proof. 

Lemma 3.2 states that the chain rule terms for characterizing Problem 2 are tied to _p_ ( _·_ ). Therefore, to characterize _G_<sup>�(</sup> _Q_<sup>_A_),</sup><sup>_G_�(</sup> _Q_<sup>_B_)</sup> (i.e., the approximations of _G_<sup>(</sup> _Q_<sup>_A_),</sup><sup>_G_(</sup> _Q_<sup>_B_)), we need to approximate the</sup> functions _f_ ( _·_ ), _q_ ( _·_ ), _c_ ( _·_ ), and hence _p_ ( _·_ ) with precision guarantees. To do so, it is convenient to consider the following decomposition of _p_ ( _·_ ). 

**Definition 3.6** (Decomposition of _p_ ( _·_ )) **.** For every _<u>j</u> ∈_ [ _L_ ], we define _p_ 1( _<u>W</u>_ <u>)</u> _<u>j, p</u>_ 2( _<u>W</u>_ <u>)</u> _<u>j</u> ∈_ R<sup>_L_</sup> as 



such that _p_ <u>(</u> _<u>W</u>_ <u>) =</u> _p_ 1( _<u>W</u>_ <u>)</u> _− p_ 2( _<u>W</u>_ <u>).</u> 

**Overview of Our Proof Strategy.** Definition 3.6 motivates the following strategy: term-by-term approximation for precision-guaranteed, almost linear time algorithms to compute (3.6) (Problem 2). 

- **Step 1.** Prove the existence of almost linear approximation algorithms for _f_ ( _·_ ) _, q_ ( _·_ ) _, c_ ( _·_ ) via low-rank approximation: Lemma 3.3, Lemma 3.5 and Lemma 3.4. 

- **Step 2.** Prove the existence of almost linear approximation algorithms for _p_ 1( _·_ ) _, p_ 2( _·_ ) and hence _p_ ( _·_ ) via the low-rank-preserving property of the multiplication between _f_ ( _·_ ) and _q_ ( _·_ ): Lemma 3.6 and Lemma 3.7. 

- **Step 3.** Prove existence of almost linear approximation algorithms for the LoRA adapter gradients (i.e., _∂∂ALQ_<sup>and</sup> _∂∂BLQ_<sup>in (3.6)) with results from</sup><sup>**Step 1 & 2**:Theorem 3.1.</sup> 

**Step 1.** We start with low-rank approximations for _f_ ( _·_ ) _, q_ ( _·_ ) _, c_ ( _·_ ). 

**Lemma 3.3** (Approximate _f_ ( _·_ ), Modified from (Alman and Song, 2023)) **.** Let Γ = _o_ (<sup>_√_</sup> log _L_ ) and _k_ 1 = _L_<sup>_o_(1)</sup> . Let _C_<sup>(1)</sup> _, C_<sup>(2)</sup> _∈_ R<sup>_L×d_</sup> , _W ∈_ R<sup>_d×d_</sup> , and _f_ <u>(</u> _<u>W</u>_ <u>)</u> = _D_<sup>_−_1</sup> exp � _C_<sup>(1)</sup> _W_ � _C_<sup>(2)�</sup><sup>_⊤_�</sup> with _D_ = diag �exp � _C_<sup>(1)</sup> _W_ � _C_<sup>(2)�</sup><sup>_⊤_�</sup> 1 _L_ � follows Definitions 3.2 to 3.5. If max ��� _C_ (1) _W_ �� _∞_<sup>_≤_</sup> 

7 

Published as a conference paper at ICLR 2025 

Γ,�� _C_ (2)�� _∞_ � _≤_ Γ, then there exist two matrices _U_ 1 _, V_ 1 _∈_ R<sup>_L×k_1</sup> such that �� _U_ 1 _V_ 1 _⊤_<sup>_−f_</sup><sup><u>(</u></sup><sup>_<u>W</u>_</sup> <u>)��</u> _∞_<sup>_≤_</sup> _ϵ/_ poly( _L_ ). In addition, it takes _L_<sup>1+</sup><sup>_o_(1)</sup> time to construct _U_ 1 and _V_ 1. 

_Proof._ This lemma is an application of (Alman and Song, 2023, Theorem 3.8). 

**Lemma 3.4** (Approximate _c_ ( _·_ )) **.** Assume all numerical values are in _O_ (log _L_ ) bits. Let _d_ = _O_ (log _L_ ) and _c_ <u>(</u> _<u>W</u>_ <u>)</u> _∈_ R<sup>_L×d_</sup> follows Definition 3.5. There exist two matrices _U_ 1 _, V_ 1 _∈_ R<sup>_L×k_1</sup> such that _⊤ U_ 1 _V_ 1<sup>_C_(3)</sup><sup>_−Y−c_</sup><sup><u>(</u></sup><sup>_<u>W</u>_</sup> <u>)</u> ��� ��� _∞_<sup>_≤ϵ/_poly(</sup><sup>_L_)</sup><sup>_._</sup> 

_Proof._ See Appendix C.3 for a detailed proof. 

**Lemma 3.5** (Approximate _q_ ( _·_ )) **.** Let _k_ 2 = _L_<sup>_o_(1)</sup> , _c_ ( _W_ ) _∈_ R<sup>_L×d_</sup> follows Definition 3.5 and let _q_ <u>(</u> _<u>W</u>_ <u>)</u> := _C_<sup>(3)</sup> ( _c_ <u>(</u> _<u>W</u>_ <u>))</u><sup>T</sup> _∈_ R<sup>_L×L_</sup> follows Lemma 3.2. There exist two matrices _U_ 2 _, V_ 2 _∈_ R<sup>_L×k_2</sup> such that �� _U_ 2 _V_ 2 _⊤_<sup>_−q_</sup><sup><u>(</u></sup><sup>_<u>W</u>_</sup> <u>)��</u> _∞_<sup>_≤ϵ/_poly(</sup><sup>_L_).In addition, it takes</sup><sup>_L_1+</sup><sup>_o_(1)time to construct</sup><sup>_U_2</sup><sup>_, V_2.</sup> 

_Proof._ See Appendix C.4 for a detailed proof. 

**Step 2.** Now, we use above lemmas to construct low-rank approximations for _p_ 1( _·_ ) _, p_ 2( _·_ ) _, p_ ( _·_ ). 

**Lemma 3.6** (Approximate _p_ 1( _·_ )) **.** Let _k_ 1 _, k_ 2 _, k_ 3 = _L_<sup>_o_(1)</sup> . Suppose _U_ 1 _, V_ 1 _∈_ R<sup>_L×k_1</sup> approximates _f_ <u>(</u> _<u>W</u>_ <u>)</u> _∈_ R<sup>_L×L_</sup> such that �� _U_ 1 _V_ 1 _⊤_<sup>_−f_</sup><sup><u>(</u></sup><sup>_<u>W</u>_</sup> <u>)��</u> _∞_<sup>_≤ϵ/_poly(</sup><sup>_L_),and</sup><sup>_U_2</sup><sup>_, V_2</sup><sup>_∈_R</sup><sup>_L×k_2approximates</sup> the _q_ <u>(</u> _<u>W</u>_ <u>)</u> _∈_ R<sup>_L×L_</sup> such that �� _U_ 2 _V_ 2 _⊤_<sup>_−q_</sup><sup><u>(</u></sup><sup>_<u>W</u>_</sup> <u>)��</u> _∞_<sup>_≤ϵ/_poly(</sup><sup>_L_).Thenthereexisttwomatrices</sup> _U_ 3 _, V_ 3 _∈_ R<sup>_L×k_3</sup> such that �� _U_ 3 _V_ 3 _⊤_<sup>_−p_1(</sup><sup>_<u>W</u>_</sup> <u>)��</u> _∞_<sup>_≤ϵ/_poly(</sup><sup>_L_)</sup><sup>_._</sup> In addition, it takes _L_<sup>1+</sup><sup>_o_(1)</sup> time to construct _U_ 3 _, V_ 3. 

_Proof Sketch._ By tensor formulation, we construct _U_ 3, _V_ 3 as tensor products of _U_ 1 _, V_ 1 and _U_ 2 _, V_ 2, respectively, while preserving their low-rank structure. Then, we show the low-rank approximation of _p_ 1( _·_ ) with bounded error by Lemma 3.3 and Lemma 3.5. See Appendix C.5 for a detailed proof. 

**Lemma 3.7** (Approximate _p_ 2( _·_ )) **.** Let _k_ 1 _, k_ 2 _, k_ 4 = _L_<sup>_o_(1)</sup> . Let _p_ 2( _<u>W</u>_ <u>)</u> _∈_ R<sup>_L×L_</sup> follow Definition 3.6 such that its _<u>j</u>_ -th column is _p_ 2( _<u>W</u>_ <u>)</u> _<u>j</u>_ = _f_ <u>(</u> _<u>W</u>_ <u>)</u> _<u>jf</u>_ <u>(</u> _<u>W</u>_ <u>)</u><sup>_⊤_</sup> _<u>j</u> q_ <u>(</u> _<u>W</u>_ <u>)</u> _<u>j</u>_ for each _<u>j</u> ∈_ [ _L_ ]. Suppose _U_ 1 _, V_ 1 _∈_ R<sup>_L×k_1</sup> approximates the f(X) such that <u>��</u> _U_ 1 _V_ 1 _⊤_<sup>_−f_</sup><sup><u>(</u></sup><sup>_<u>W</u>_</sup> <u>)��</u> _∞_<sup>_≤ϵ/_poly(</sup><sup>_L_),and</sup><sup>_U_2</sup><sup>_, V_2</sup><sup>_∈_R</sup><sup>_L×k_2</sup> approximates the _q_ <u>(</u> _<u>W</u>_ <u>)</u> _∈_ R<sup>_L×L_</sup> such that �� _U_ 2 _V_ 2 _⊤_<sup>_−q_</sup><sup><u>(</u></sup><sup>_<u>W</u>_</sup> <u>)��</u> _∞_<sup>_≤ϵ/_poly(</sup><sup>_L_).Thenthereexist</sup> matrices _U_ 4 _, V_ 4 _∈_ R<sup>_L×k_4</sup> such that �� _U_ 4 _V_ 4 _⊤_<sup>_−p_2(</sup><sup>_<u>W</u>_</sup> <u>)��</u> _∞_<sup>_≤ϵ/_poly(</sup><sup>_L_)</sup> In addition, it takes _L_<sup>1+</sup><sup>_o_(1)</sup> time to construct _U_ 4 _, V_ 4. 

_Proof Sketch._ By considering the following decomposition through tensor formulation 



we approximate the _p_ 2( _·_ ) part by part. Specifically, for (I), we show its low-rank approximation by observing the low-rank-preserving property of the multiplication between _f_ ( _·_ ) and _q_ ( _·_ ) (from Lemma 3.3 and Lemma 3.5). For (II), we show its low-rank approximation by the low-rank structure of _f_ ( _·_ ) and (I). See Appendix C.6 for a detailed proof. 

8 

Published as a conference paper at ICLR 2025 

**Step 3.** Combining above, we arrive our main result: almost linear algorithm for Problem 2. 

**Theorem 3.1** (Main Result: Existence of Almost Linear Time ALoRAGC) **.** Suppose all numerical values are in _O_ (log _L_ )-bits encoding. Recall that _W_ = _W_<sup>_⋆_</sup> _Q_<sup>+</sup><sup>_BQAQ∈_R</sup><sup>_d×d_with</sup> _W_<sup>_⋆_</sup> _Q_<sup>:=</sup><sup>_rW ⋆_</sup> _Q_<sup>_/α_.Let</sup><sup>_C_(1)=</sup><sup>_X_(</sup><sup>_Q_)</sup><sup>_<u>α</u>_</sup> _r_<sup>_, C_(2)=</sup><sup>_X_(</sup><sup>_K_)</sup><sup>_W ⋆_</sup> _K_<sup>follows(3.2).If</sup> �� _C_ (1) _W_ �� _∞_<sup>_≤_Γ</sup> and �� _C_ (2)�� _∞_<sup>_≤_Γ,whereΓ=</sup><sup>_o_(</sup><sup>_√_</sup> log _L_ ), then there exists a _L_<sup>1+</sup><sup>_o_(1)</sup> time algorithm to solve ALoRAGC � _L, d_ = _O_ (log _L_ ) _, r_ = _L_<sup>_o_(1)</sup> _, ϵ_ = 1 _/_ poly( _L_ )� (i.e., Problem 2). In particular, this algorithm outputs gradient matrices _G_<sup>�(</sup> _Q_<sup>_A_)</sup> _∈_ R<sup>_d×r_</sup> _, G_<sup>�(</sup> _Q_<sup>_B_)</sup> _∈_ R<sup>_r×d_</sup> such that ( _A_ ) ( _B_ ) _∥ ∂_<sup>_∂_</sup> _<u>A</u>_<sup>_L_</sup> _−_ _<u>G</u>_<sup>�</sup> _~~Q~~_<sup>_∥∞≤_1</sup><sup>_/_poly(</sup><sup>_L_)</sup><sup>_,_</sup> and _∥ ∂_<sup>_∂_</sup> _<u>B</u>_<sup>_L_</sup> _−_ _<u>G</u>_<sup>�</sup> _~~Q~~_<sup>_∥∞≤_1</sup><sup>_/_poly(</sup><sup>_L_)</sup><sup>_._</sup> _~~Q Q~~_ 

_Proof Sketch._ By Lemma 3.2, we have<sup>_∂L_</sup> _/∂AQ_<sup>=vec(</sup><sup>_B_</sup> _Q_<sup>_⊤_(</sup><sup>_C_(1))</sup><sup>_⊤p_</sup><sup><u>(</u></sup><sup>_<u>W</u>_</sup> <u>)</u> _C_<sup>(2)</sup> ), and<sup>_∂L_</sup> _/∂B_ _~~Q~~_<sup>=</sup> vec(( _C_<sup>(1)</sup> )<sup>_⊤_</sup> _p_ <u>(</u> _<u>W</u>_ <u>)</u> _AQC_<sup>(2)</sup> ). By Lemma 3.2 and Definition 3.6, we have _p_ <u>(</u> _<u>W</u>_ <u>)</u> = _p_ 1( _<u>W</u>_ <u>)</u> _− p_ 2( _<u>W</u>_ <u>).</u> Firstly, we notice that the _exact_ computation of _BQ_<sup>_⊤_(</sup><sup>_C_(1))and</sup><sup>_AQC_(2)takesonly</sup> _L_<sup>1+</sup><sup>_o_(1)</sup> time, by _AQ ∈_ R<sup>_r×d_</sup> , _BQ ∈_ R<sup>_d×r_</sup> , _C_<sup>(1)</sup> , _C_<sup>(2)</sup> _∈_ R<sup>_L×d_</sup> . Thus, to show the existence of _L_<sup>1+</sup><sup>_o_(1)</sup> time algorithms for Problem 2, we prove fast low-rank approximations for _BQ_<sup>_⊤_(</sup><sup>_C_(1))</sup><sup>_⊤p_1(</sup><sup>_<u>W</u>_</sup> <u>)</u> _C_<sup>(2)</sup> and ( _C_<sup>(1)</sup> )<sup>_⊤_</sup> _p_ 1( _<u>W</u>_ <u>)</u> _AQC_<sup>(2)</sup> by Lemma 3.6. The fast low-rank approximations for _−BQ_<sup>_⊤_(</sup><sup>_C_(1))</sup><sup>_⊤p_2(</sup><sup>_<u>W</u>_</sup> <u>)</u> _C_<sup>(2)</sup> and _−_ ( _C_<sup>(1)</sup> )<sup>_⊤_</sup> _p_ 2( _<u>W</u>_ <u>)</u> _AQC_<sup>(2)</sup> follow trivially. See Appendix C.7 for a detailed proof. 

**General Case: Full LoRA Adaptation on** _WK, WQ, WV_ **.** We defer the analysis of full LoRA on transformer ((C2) **General Case:** adapting both _WK, WQ, WV_ ) to Appendix A due to page limit. Importantly, we also prove the existence of an almost linear-time LoRA (Theorem A.1). In addition, we derive the norm bound conditions required for it to hold. 

### 4 NORM-BASED PHASE TRANSITION IN EFFICIENCY 

In this section, we characterize the computational limits of all possible efficient algorithms of ALoRAGC, via fine-grained reduction under the Strong Exponential Time Hypothesis (SETH). 

**Strong Exponential Time Hypothesis (SETH).** Impagliazzo and Paturi (2001) introduce the Strong Exponential Time Hypothesis (SETH) as a stronger form of the `P` = `NP` conjecture. It suggests that our current best `SAT` algorithms are optimal and is a popular conjecture for proving fine-grained lower bounds for a wide variety of algorithmic problems (Williams, 2018b; 2013; Cygan et al., 2016). 

**Hypothesis 1** (SETH) **.** For every _ϵ >_ 0, there is a positive integer _k ≥_ 3 such that _k_ - `SAT` on formulas with _n_ variables cannot be solved in _O_ (2<sup>(1</sup><sup>_−ϵ_)</sup><sup>_n_</sup> ) time, even by a randomized algorithm. 

Our primary technique involves casting the ALoRAGC problem (Problem 1) as a fine-grained reduction under SETH, from the hardness result of fast attention approximation algorithm (Alman and Song, 2023). For simplicity of analysis, we consider the special case (C1). 

**Theorem 4.1** (Inefficient Threshold) **.** Let _κ_ : N _→_ N by any function with _κ_ ( _L_ ) = _ω_ (1) and _κ_ ( _L_ ) = _o_ (log _L_ ). Let Γ = _O_ (<sup>_√_</sup> log _L · κ_ ( _L_ )). Assuming Hypothesis 1, there is no algorithm running in time _O_ ( _L_<sup>2</sup><sup>_−δ_</sup> ) for any constant _δ >_ 0 for ALoRAGC( _L, d_ = _O_ (log _L_ ) _, r < d, ϵ_ ), i.e., Problem 2, subject to (3.3), even in the case where the input and weight matrices satisfy _∥X_<sup>(</sup><sup>_K_)</sup> _WK_<sup>_⋆∥∞≤_Γ,</sup> _∥αXi_<sup>(</sup><sup>_Q_)</sup> _BQAQ/r∥∞ ≤_ Γ, _Y_ = 0 and _ϵ_ = _O_ ((log _L_ )<sup>_−_4</sup> ). 

_Proof Sketch._ Firstly, we recall the hardness of sub-quadratic **Att** ention **G** radient **C** omputation approximation, i.e., AttLGC from (Alman and Song, 2024a) (defined in Definition E.1). This serves as a reference point for the complexity we anticipate for ALoRAGC defined in Problem 2. We then proceed with a reduction from problem AttLGC to problem ALoRAGC. Essentially, by showing that AttLGC is at least as hard as ALoRAGC, and then showing how to solve AttLGC using a solution to ALoRAGC, we establish the hardness of ALoRAGC. See for Appendix E for a detailed proof. 

9 

Published as a conference paper at ICLR 2025 

**Remark 4.1.** Theorem 4.1 suggests an efficiency threshold for Γ. Only below this threshold are efficient algorithms for ALoRAGC possible. This is a Γ-based phase transition behavior in efficiency. 

**Remark 4.2.** In Theorem 4.1, we show that even the simplest single-data-point case with _Y_ = 0 is hard. Hence, our result also applies to the special case (C1) (i.e., Problem 2) and general case (C2) (i.e., Problem 3). Specifically, it is evident that computing the gradient for multiple data points (whether the full gradient or a stochastic mini-batch gradient) is _at least_ as hard as for a single data point. The hardness follows trivially. 

### 5 DISCUSSION AND CONCLUDING REMARKS 

We study the computational limits of the Low-Rank Adaptation (LoRA) for transformer-based model finetuning using fine-grained complexity theory (i.e., under Hypothesis 1). Our main contribution is the proof of the existence of almost linear approximation algorithms for LoRA adaptation on transformer-based models. We accomplish this by utilizing the hierarchical low-rank structures of LoRA gradients (Lemmas 3.3 to 3.5) and approximating the gradients with a series of chained low-rank approximations (Lemmas 3.6 and 3.7). To showcase our theory, we establish such almost linear approximation for both partial (Theorem 3.1) and full LoRA adaptions (Theorem A.1) of attention weights. In addition, we identify a phase transition behavior in the efficiency of all possible variants of LoRA (Theorem 4.1) by adjusting the norm upper-bound Γ of input, pretrained, and adaptor weights. Specifically, we establish an “inefficiency threshold” for Γ, only below which adapting transformer-based models with LoRA in _L_<sup>2</sup><sup>_−o_(1)</sup> (sub-quadratic) time is possible. 

**Remark 5.1** (General Case: Full LoRA Adaptation on _WK, WQ, WV_ ) **.** We defer the analysis of full LoRA on transformer (adapting both _WK, WQ, WV_ matrices) to Appendix A due to page limit. 

**Remark 5.2** (Insights for Practitionars: Necessary Conditions for Efficient and Robust LoRA) **.** This work is about LoRA on transformer models. Therefore, the computational bottleneck is by design _O_ ( _L_<sup>2</sup> ) (see Appendix F for discussions and a proof.) In this regard, our work provides in-depth analysis to address this _O_ ( _L_<sup>2</sup> ) bottleneck and provides useful insights and guidance for designing efficient LoRA algorithms and methods with precision guarantees: 

- **Theorem 4.1: Necessary Conditions for Subqudratic Time LoRA.** Proper normalization of the composed norms, e.g., _∥X_<sup>(</sup><sup>_K_)</sup> _WK_<sup>_⋆∥≤_Γ and</sup><sup>_∥αX_</sup> _i_<sup>(</sup><sup>_Q_)</sup> _BQAQ/r∥≤_ Γ with Γ = _O_ (<sup>_√_</sup> log _L·κ_ ( _L_ )). 

- **Theorems 3.1 and A.1: Necessary Conditions for Almost Linear Time LoRA.** Proper normalization of the composed norms, e.g., 

- For partial LoRA on _WQ, WV_ (Theorem 3.1): �� _<u>αr</u>_<sup>_X_(</sup><sup>_Q_)</sup><sup>_W_</sup> �� _∞_<sup>_≤_Γand</sup> �� _X_ ( _K_ ) _W ⋆K_ �� _∞_<sup>_≤_Γ</sup> with Γ = _o_ (<sup>_√_</sup> log _L_ ). 

- For full LoRA on _WK, WQ, WV_ (Theorem A.1): �� _X_ ( _Q_ ) � _WQ_<sup>_⋆_+</sup><sup>_<u>α</u>_</sup> _r_<sup>_BQAQ_</sup> � _WK_ �� _∞_<sup>_≤_Γ,</sup> �� _X_ ( _K_ )�� _≤_ Γ, �� _X_ ( _Q_ ) _WQ_ �� _≤_ Γ, and �� _X_ ( _K_ ) � _WK_<sup>_⋆_+</sup><sup>_<u>α</u>_</sup> _r_<sup>_BKAK_</sup> ��� _∞_<sup>_≤_Γ with Γ =</sup><sup>_o_(</sup><sup>_√_</sup> log _L_ ). 

Suitable normalization of the composed norms can be implemented using pre-activation layer normalization (Xiong et al., 2020; Wang et al., 2019) to control _∥X∥_ , or outlier-removing attention activation functions (Hu et al., 2024a) to control _{∥Wµ∥, ∥Aµ∥, ∥Bµ∥}µ_ = _K,Q_ . On one hand, our findings provide formal justifications for these methods. On the other hand, these necessary conditions also motivate the design of future efficient methods with minimal model and data assumptions. 

**Remark 5.3** (Self- and Cross-Attention) **.** We emphasize that all these results hold for not only selfattention but also cross-attention due to our generic problem setting (Definition 2.5 and Remark 2.1). 

**Proof-of-Concept Experiments.** We provide numerical results to justify our theory in Appendix G. 

**Limitations.** We identify necessary conditions for fast LoRA methods, not sufficient conditions. Therefore, our results do not lead to direct implementations. This limitation is inherent to hardness results (Toolkit, 2013). However, as discussed above, we expect our findings to provide valuable insights for future efficient LoRA implementations in both forward and backward computations. 

**Impact Statement.** This theoretical work aims to elucidate the foundations of large transformer-based foundation models and is not expected to have negative social impacts. 

**Related Works.** We defer the discussion of related works to Appendix B due to page limit. 

10 

Published as a conference paper at ICLR 2025 

### ACKNOWLEDGMENTS 

JH thanks Mimi Gallagher, Sara Sanchez, Dino Feng, and Andrew Chen for enlightening discussions; Yen-Ju Lu, Shang Wu, Robin Luo, and Jiahao Yu for collaborations on related topics; Shang Wu and authors of (Wu et al., 2024c) for assistance with numerical experiments; and the Red Maple Family for their support. The authors also thank the anonymous reviewers and program chairs for their constructive comments. 

JH is partially supported by the Walter P. Murphy Fellowship. HL is partially supported by NIH R01LM1372201, AbbVie and Dolby. EJK thanks the National Center for Theoretical Sciences of Taiwan for funding (112-2124-M-002-003). This research was supported in part through the computational resources and staff contributions provided for the Quest high performance computing facility at Northwestern University which is jointly supported by the Office of the Provost, the Office for Research, and Northwestern University Information Technology. The content is solely the responsibility of the authors and does not necessarily represent the official views of the funding agencies. 

### REFERENCES 

- Amir Abboud, Virginia Vassilevska Williams, and Oren Weimann. Consequences of faster alignment of sequences. In _Automata, Languages, and Programming: 41st International Colloquium, ICALP 2014, Copenhagen, Denmark, July 8-11, 2014, Proceedings, Part I 41_ , pages 39–51. Springer, 2014. 

- Amir Abboud, Arturs Backurs, Thomas Dueholm Hansen, Virginia Vassilevska Williams, and Or Zamir. Subtree isomorphism revisited. _ACM Transactions on Algorithms (TALG)_ , 14(3):1–23, 2018. 

- Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- Amol Aggarwal and Josh Alman. Optimal-degree polynomial approximations for exponentials and gaussian kernel density estimation. In _Proceedings of the 37th Computational Complexity Conference_ , CCC ’22, Dagstuhl, DEU, 2022. Schloss Dagstuhl–Leibniz-Zentrum fuer Informatik. ISBN 9783959772419. doi: 10.4230/LIPIcs.CCC.2022.22. 

- Josh Alman and Zhao Song. Fast attention requires bounded entries. In _Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS)_ , 2023. 

- Josh Alman and Zhao Song. The fine-grained complexity of gradient computation for training large language models. _arXiv preprint arXiv:2402.04497_ , 2024a. 

- Josh Alman and Zhao Song. How to capture higher-order correlations? generalizing matrix softmax attention to kronecker computation. In _ICLR_ . arXiv preprint arXiv:2310.04064, 2024b. 

- Josh Alman and Hantao Yu. Fundamental limitations on subquadratic alternatives to transformers. _arXiv preprint arXiv:2410.04271_ , 2024. 

- Josh Alman, Timothy Chu, Aaron Schild, and Zhao Song. Algorithms and hardness for linear algebra on geometric graphs. In _2020 IEEE 61st Annual Symposium on Foundations of Computer Science (FOCS)_ , pages 541–552. IEEE, 2020. 

- Arturs Backurs and Piotr Indyk. Which regular expression patterns are hard to match? In _2016 IEEE 57th Annual Symposium on Foundations of Computer Science (FOCS)_ , pages 457–466. IEEE, 2016. 

- Arturs Backurs, Piotr Indyk, and Ludwig Schmidt. On the fine-grained complexity of empirical risk minimization: Kernel methods and neural networks. _Advances in Neural Information Processing Systems_ , 30, 2017. 

- Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. _arXiv preprint arXiv:2108.07258_ , 2021. 

11 

Published as a conference paper at ICLR 2025 

Yelysei Bondarenko, Markus Nagel, and Tijmen Blankevoort. Quantizable transformers: Removing outliers by helping attention heads do nothing. _Advances in Neural Information Processing Systems (NeurIPS)_ , 36, 2023. 

- Karl Bringman and Marvin Künnemann. Multivariate fine-grained complexity of longest common subsequence. In _Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium on Discrete Algorithms_ , pages 1216–1235. SIAM, 2018. 

- Karl Bringmann. Why walking the dog takes time: Frechet distance has no strongly subquadratic algorithms unless seth fails. In _2014 IEEE 55th Annual Symposium on Foundations of Computer Science_ , pages 661–670. IEEE, 2014. 

- Karl Bringmann and Wolfgang Mulzer. Approximability of the discrete fréchet distance. _Journal of Computational Geometry_ , 7(2):46–76, 2016. 

- Karl Bringmann, Allan Grønlund, and Kasper Green Larsen. A dichotomy for regular expression membership testing. In _2017 IEEE 58th Annual Symposium on Foundations of Computer Science (FOCS)_ , pages 307–318. IEEE, 2017. 

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901, 2020. 

- Kevin Buchin, Maike Buchin, Maximilian Konzack, Wolfgang Mulzer, and André Schulz. Finegrained analysis of problems on curves. _EuroCG, Lugano, Switzerland_ , 3, 2016. 

- Chris Calabro, Russell Impagliazzo, and Ramamohan Paturi. The complexity of unique k-sat: An isolation lemma for k-cnfs. In _International Workshop on Parameterized and Exact Computation_ , pages 47–56. Springer, 2009. 

- Timothy M Chan, Virginia Vassilevska Williams, and Yinzhan Xu. Hardness for triangle problems under even more believable hypotheses: reductions from real apsp, real 3sum, and ov. In _Proceedings of the 54th Annual ACM SIGACT Symposium on Theory of Computing_ , pages 1501–1514, 2022. 

- Lijie Chen. On the hardness of approximate and exact (bichromatic) maximum inner product. In _Proceedings of the 33rd Computational Complexity Conference_ , pages 1–45, 2018. 

- Lijie Chen and Ryan Williams. An equivalence class for orthogonal vectors. In _Proceedings of the Thirtieth Annual ACM-SIAM Symposium on Discrete Algorithms_ , pages 21–40. SIAM, 2019. 

- Marek Cygan, Holger Dell, Daniel Lokshtanov, Dániel Marx, Jesper Nederlof, Yoshio Okamoto, Ramamohan Paturi, Saket Saurabh, and Magnus Wahlström. On problems as hard as cnf-sat. _ACM Transactions on Algorithms (TALG)_ , 12(3):1–24, 2016. 

- Mina Dalirrooyfard, Ray Li, and Virginia Vassilevska Williams. Hardness of approximate diameter: Now for undirected graphs. In _2021 IEEE 62nd Annual Symposium on Foundations of Computer Science (FOCS)_ , pages 1021–1032. IEEE, 2022. 

- Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Huaian Diao, Zhao Song, Wen Sun, and David Woodruff. Sketching for kronecker product regression and p-splines. In _International Conference on Artificial Intelligence and Statistics_ , pages 1299– 1308. PMLR, 2018. 

- Huaian Diao, Rajesh Jayaram, Zhao Song, Wen Sun, and David Woodruff. Optimal sketching for kronecker product regression and low rank approximation. _Advances in neural information processing systems_ , 32, 2019. 

- Ning Ding, Yujia Qin, Guang Yang, Fuchao Wei, Zonghan Yang, Yusheng Su, Shengding Hu, Yulin Chen, Chi-Min Chan, Weize Chen, et al. Delta tuning: A comprehensive study of parameter efficient methods for pre-trained language models. _arXiv preprint arXiv:2203.06904_ , 2022. 

12 

Published as a conference paper at ICLR 2025 

- Ning Ding, Xingtai Lv, Qiaosen Wang, Yulin Chen, Bowen Zhou, Zhiyuan Liu, and Maosong Sun. Sparse low-rank adaptation of pre-trained language models. In _The 2023 Conference on Empirical Methods in Natural Language Processing_ , 2023. 

- Luciano Floridi and Massimo Chiriatti. Gpt-3: Its nature, scope, limits, and consequences. _Minds and Machines_ , 30:681–694, 2020. 

- Jiawei Gao, Russell Impagliazzo, Antonina Kolokolova, and Ryan Williams. Completeness for first-order properties on sparse structures with algorithmic applications. _ACM Transactions on Algorithms (TALG)_ , 15(2):1–35, 2018. 

- Yeqi Gao, Zhao Song, Weixin Wang, and Junze Yin. A fast optimization view: Reformulating single layer attention in llm based on tensor and svm trick, and solving it in matrix multiplication time. _arXiv preprint arXiv:2309.07418_ , 2023a. 

- Yeqi Gao, Zhao Song, and Shenghao Xie. In-context learning for attention scheme: from single softmax regression to multiple softmax regression via a tensor trick. _arXiv preprint arXiv:2307.02419_ , 2023b. 

- Jiuxiang Gu, Yingyu Liang, Heshan Liu, Zhenmei Shi, Zhao Song, and Junze Yin. Conv-basis: A new paradigm for efficient attention inference and gradient computation in transformers. _arXiv preprint arXiv:2405.05219_ , 2024a. 

- Jiuxiang Gu, Yingyu Liang, Zhenmei Shi, Zhao Song, and Yufa Zhou. Tensor attention training: Provably efficient learning of higher-order transformers. _arXiv preprint arXiv:2405.16411_ , 2024b. 

- Han Guo, Philip Greengard, Eric Xing, and Yoon Kim. LQ-loRA: Low-rank plus quantized matrix decomposition for efficient language model finetuning. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- Soufiane Hayou, Nikhil Ghosh, and Bin Yu. Lora+: Efficient low rank adaptation of large models. _arXiv preprint arXiv:2402.12354_ , 2024. 

- Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. _arXiv preprint arXiv:2106.09685_ , 2021. 

- Jerry Yao-Chieh Hu, Donglin Yang, Dennis Wu, Chenwei Xu, Bo-Yu Chen, and Han Liu. On sparse modern hopfield model. In _Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS)_ , 2023. 

- Jerry Yao-Chieh Hu, Pei-Hsuan Chang, Robin Luo, Hong-Yu Chen, Weijian Li, Wei-Po Wang, and Han Liu. Outlier-efficient hopfield layers for large transformer-based models. In _Forty-first International Conference on Machine Learning (ICML)_ , 2024a. 

Jerry Yao-Chieh Hu, Bo-Yu Chen, Dennis Wu, Feng Ruan, and Han Liu. Nonparametric modern hopfield models. _arXiv preprint arXiv:2404.03900_ , 2024b. 

- Jerry Yao-Chieh Hu, Thomas Lin, Zhao Song, and Han Liu. On computational limits of modern hopfield models: A fine-grained complexity analysis. In _Forty-first International Conference on Machine Learning (ICML)_ , 2024c. 

- Jerry Yao-Chieh Hu, Dennis Wu, and Han Liu. Provably optimal memory capacity for modern hopfield models: Transformer-compatible dense associative memories as spherical codes. _Advances in Neural Information Processing Systems_ , 37:70693–70729, 2025. 

- Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu Pang, Chao Du, and Min Lin. Lorahub: Efficient cross-task generalization via dynamic lora composition. _arXiv preprint arXiv:2307.13269_ , 2023. 

- Russell Impagliazzo and Ramamohan Paturi. On the complexity of k-sat. _Journal of Computer and System Sciences_ , 62(2):367–375, 2001. 

13 

Published as a conference paper at ICLR 2025 

Yanrong Ji, Zhihan Zhou, Han Liu, and Ramana V Davuluri. Dnabert: pre-trained bidirectional encoder representations from transformers model for dna-language in genome. _Bioinformatics_ , 37 (15):2112–2120, 2021. 

- Jacob Kahn, Morgane Riviere, Weiyi Zheng, Evgeny Kharitonov, Qiantong Xu, Pierre-Emmanuel Mazaré, Julien Karadayi, Vitaliy Liptchinsky, Ronan Collobert, Christian Fuegen, et al. Libri-light: A benchmark for asr with limited or no supervision. In _ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pages 7669–7673. IEEE, 2020. 

- CS Karthik and Pasin Manurangsi. On closest pair in euclidean metric: Monochromatic is as hard as bichromatic. _Combinatorica_ , 40(4):539–573, 2020. 

- Robert Krauthgamer and Ohad Trabelsi. Conditional lower bounds for all-pairs max-flow. _ACM Transactions on Algorithms (TALG)_ , 14(4):1–15, 2018. 

- Yinghui Li, Jing Yang, and Jiliang Wang. Dylora: Towards energy efficient dynamic lora transmission control. In _IEEE INFOCOM 2020-IEEE Conference on Computer Communications_ , pages 2312– 2320. IEEE, 2020. 

- Yixiao Li, Yifan Yu, Chen Liang, Nikos Karampatziakis, Pengcheng He, Weizhu Chen, and Tuo Zhao. Loftq: LoRA-fine-tuning-aware quantization for large language models. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, KwangTing Cheng, and Min-Hung Chen. Dora: Weight-decomposed low-rank adaptation. _arXiv preprint arXiv:2402.09353_ , 2024. 

- Soumi Maiti, Yifan Peng, Shukjae Choi, Jee-weon Jung, Xuankai Chang, and Shinji Watanabe. Voxtlm: Unified decoder-only models for consolidating speech recognition, synthesis and speech, text continuation tasks. In _ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pages 13326–13330. IEEE, 2024. 

- Yuren Mao, Yuhang Ge, Yijiang Fan, Wenyi Xu, Yu Mi, Zhonghao Hu, and Yunjun Gao. A survey on lora of large language models. _Frontiers of Computer Science_ , 19(7):197605, 2025. 

- Michael Moor, Oishi Banerjee, Zahra Shakeri Hossein Abad, Harlan M Krumholz, Jure Leskovec, Eric J Topol, and Pranav Rajpurkar. Foundation models for generalist medical artificial intelligence. _Nature_ , 616(7956):259–265, 2023. 

- Eric Nguyen, Michael Poli, Marjan Faizi, Armin Thomas, Michael Wornow, Callum Birch-Sykes, Stefano Massaroli, Aman Patel, Clayton Rabideau, Yoshua Bengio, et al. Hyenadna: Long-range genomic sequence modeling at single nucleotide resolution. _Advances in neural information processing systems_ , 36, 2024. 

- Rui Pan, Xiang Liu, Shizhe Diao, Renjie Pi, Jipeng Zhang, Chi Han, and Tong Zhang. Lisa: Layerwise importance sampling for memory-efficient large language model fine-tuning. _arXiv preprint arXiv:2403.17919_ , 2024. 

- Liam Roditty and Virginia Vassilevska Williams. Fast approximation algorithms for the diameter and radius of sparse graphs. In _Proceedings of the forty-fifth annual ACM symposium on Theory of computing_ , pages 515–524, 2013. 

- Aviad Rubinstein. Hardness of approximate nearest neighbor search. In _Proceedings of the 50th annual ACM SIGACT symposium on theory of computing (STOC)_ , pages 1260–1268, 2018. 

- Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al. Large language models encode clinical knowledge. _Nature_ , 620(7972):172–180, 2023. 

- Mingjie Sun, Xinlei Chen, J Zico Kolter, and Zhuang Liu. Massive activations in large language models. _arXiv preprint arXiv:2402.17762_ , 2024. 

14 

Published as a conference paper at ICLR 2025 

- Arun James Thirunavukarasu, Darren Shu Jeng Ting, Kabilan Elangovan, Laura Gutierrez, Ting Fang Tan, and Daniel Shu Wei Ting. Large language models in medicine. _Nature medicine_ , 29(8): 1930–1940, 2023. 

- A Theorist’s Toolkit. Lecture 24: Hardness assumptions. 2013. 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ , 2023a. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ , 2023b. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 

- Qiang Wang, Bei Li, Tong Xiao, Jingbo Zhu, Changliang Li, Derek F Wong, and Lidia S Chao. Learning deep transformer models for machine translation. _arXiv preprint arXiv:1906.01787_ , 2019. 

- Ryan Williams. Finding paths of length k in _o_<sup>_∗_</sup> (2<sup>_k_</sup> ) time. _Information Processing Letters_ , 109(6): 315–318, 2013. 

- Ryan Williams. On the difference between closest, furthest, and orthogonal pairs: Nearly-linear vs barely-subquadratic complexity. In _Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium on Discrete Algorithms_ , pages 1207–1215. SIAM, 2018a. 

- Virginia Vassilevska Williams. On some fine-grained questions in algorithms and complexity. In _Proceedings of the international congress of mathematicians: Rio de janeiro 2018_ , pages 3447– 3487. World Scientific, 2018b. 

- Dennis Wu, Jerry Yao-Chieh Hu, Teng-Yun Hsiao, and Han Liu. Uniform memory retrieval with larger capacity for modern hopfield models. In _Forty-first International Conference on Machine Learning (ICML)_ , 2024a. 

- Dennis Wu, Jerry Yao-Chieh Hu, Weijian Li, Bo-Yu Chen, and Han Liu. STanhop: Sparse tandem hopfield model for memory-enhanced time series prediction. In _The Twelfth International Conference on Learning Representations (ICLR)_ , 2024b. 

- Shang Wu, Yen-Ju Lu, Haozheng Luo, Jerry Yao-Chieh Hu, Jiayi Wang, Najim Dehak, Jesus Villalba, and Han Liu. Fast adaptation and robust quantization of multi-modal foundation models from associative memory: A case study in speechlm. In _Workshop on Efficient Systems for Foundation Models II@ ICML2024_ , 2024c. 

- Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, and Gideon Mann. Bloomberggpt: A large language model for finance. _arXiv preprint arXiv:2303.17564_ , 2023. 

- Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tieyan Liu. On layer normalization in the transformer architecture. In _International Conference on Machine Learning_ , pages 10524–10533. PMLR, 2020. 

- Chenwei Xu, Yu-Chao Huang, Jerry Yao-Chieh Hu, Weijian Li, Ammar Gilani, Hsi-Sheng Goan, and Han Liu. Bishop: Bi-directional cellular learning for tabular data with generalized sparse modern hopfield model. In _Forty-first International Conference on Machine Learning (ICML)_ , 2024a. 

- Yuhui Xu, Lingxi Xie, Xiaotao Gu, Xin Chen, Heng Chang, Hengheng Zhang, Zhengsu Chen, XIAOPENG ZHANG, and Qi Tian. QA-loRA: Quantization-aware low-rank adaptation of large language models. In _The Twelfth International Conference on Learning Representations_ , 2024b. 

- Hongyang Yang, Xiao-Yang Liu, and Christina Dan Wang. Fingpt: Open-source financial large language models. _arXiv preprint arXiv:2306.06031_ , 2023. 

15 

Published as a conference paper at ICLR 2025 

Yuchen Zeng and Kangwook Lee. The expressive power of low-rank adaptation. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. Adaptive budget allocation for parameter-efficient fine-tuning. In _The Eleventh International Conference on Learning Representations_ , 2023. 

- Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. _arXiv preprint arXiv:2205.01068_ , 2022. 

- Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, and Zheyan Luo. Llamafactory: Unified efficient fine-tuning of 100+ language models. _arXiv preprint arXiv:2403.13372_ , 2024. 

- Zhihan Zhou, Yanrong Ji, Weijian Li, Pratik Dutta, Ramana Davuluri, and Han Liu. Dnabert-2: Efficient foundation model and benchmark for multi-species genome. _arXiv preprint arXiv:2306.15006_ , 2023. 

- Zhihan Zhou, Winmin Wu, Harrison Ho, Jiayi Wang, Lizhen Shi, Ramana V Davuluri, Zhong Wang, and Han Liu. Dnabert-s: Learning species-aware dna embedding with genome foundation models. _arXiv preprint arXiv:2402.08777_ , 2024. 

- Zhihan Zhou, Robert Riley, Satria Kautsar, Weimin Wu, Rob Egan, Steven Hofmeyr, Shira GoldhaberGordon, Mutian Yu, Harrison Ho, Fengchen Liu, et al. Genomeocean: An efficient genome foundation model trained on large-scale metagenomic assemblies. _bioRxiv_ , pages 2025–01, 2025. 

16 

Published as a conference paper at ICLR 2025 

# **Appendix** 

|**A **|**General Case: Full LoRA Adaptation on**_WK_**,**_WQ_**and**_WV_|**18**|
|---|---|---|
|**B**|**Related Works**|**20**|
|**C **|**Proofs of Section 3**|**22**|
||C.1<br>Proof ofLemma 3.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . .<br>22|
||C.2<br>Proof ofLemma 3.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . .<br>23|
||C.3<br>Proof ofLemma 3.4 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . .<br>24|
||C.4<br>Proof ofLemma 3.5 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . .<br>24|
||C.5<br>Proof ofLemma 3.6 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . .<br>25|
||C.6<br>Proof ofLemma 3.7 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . .<br>26|
||C.7<br>Proof ofTheorem 3.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . .<br>27|
|**D **|**Proof of Theorem A.1**|**29**|
|**E**|**Proof of Theorem 4.1**|**37**|
|**F**|**Quadratic Time Complexity of Exact LoRA Gradient Computation**|**38**|
|**G **|**Proof-of-Concept Experiments**|**40**|



17 

Published as a conference paper at ICLR 2025 

A GENERAL CASE: FULL LORA ADAPTATION ON _WK_ , _WQ_ AND _WV_ 

Similarly, we formulate the full adaptation (C2) of an attention head as the following LoRA loss. 

**Definition A.1** (Adapting _WK_ , _WQ_ , _WV_ of Generic Attention with LoRA) **.** Let _D_ = _{_ ( _Xi_<sup>(</sup><sup>_K_)</sup> _, Xi_<sup>(</sup><sup>_Q_)</sup> _, Xi_<sup>(</sup><sup>_V_)</sup> ) _, Yi}_<sup>_N_</sup> _i_ =1<sup>beadatasetofsize</sup><sup>_N_withthetriplet</sup><sup>_X_</sup> _i_<sup>(</sup><sup>_K_)</sup> _, Xi_<sup>(</sup><sup>_Q_)</sup> _, Xi_<sup>(</sup><sup>_V_)</sup> _∈_ R<sup>_L×d_</sup> being the input and _Yi ∈_ R<sup>_L×d_</sup> being the label. The problem of fine-tuning a generic attention with LoRA with _ℓ_ 2 loss from dataset _D_ is formulated as 



By simplifications (S1), (S3) and (S4), we fix _WV_ , set _β_ =<sup>_α_</sup> _/r_ = 1 and consider LoRA adaptation on a single data point. Akin to simplification (S2), we introduce _CK_<sup>(1)</sup><sup>_, C_</sup> _K_<sup>(2)</sup><sup>_, C_</sup> _Q_<sup>(1)</sup><sup>_, C_</sup> _Q_<sup>(2)</sup><sup>_, C_(3)</sup><sup>_∈_R</sup><sup>_L×d_:</sup> 



**Remark A.1.** _CK_<sup>(1)</sup><sup>_, C_</sup> _K_<sup>(2)</sup><sup>_, C_(3) are constants with respect to adapting</sup><sup>_BK, AK_with gradient updates.</sup> _CQ_<sup>(1)</sup><sup>_, C_</sup> _Q_<sup>(2)</sup><sup>_, C_(3)are constants with respect to adapting</sup><sup>_BQ, AQ_with gradient updates.</sup> 

Therefore, the full LoRA adaptation loss in Definition A.1 becomes 



where _D_ = diag � exp � _CK_<sup>(1)(</sup><sup>_W ⋆_</sup> _K_<sup>+</sup><sup>_BKAK_)</sup><sup>_⊤_(</sup><sup>_C_</sup> _K_<sup>(2))</sup><sup>_⊤_�</sup> 1 _L_ � = diag � exp � _CQ_<sup>(1)(</sup><sup>_W ⋆_</sup> _Q_<sup>+</sup> _BQAQ_ )( _CQ_<sup>(2))</sup><sup>_⊤_�</sup> 1 _L_ � _∈_ R<sup>_L×L_</sup> . 

Similar to Section 3, we introduce the following problem to characterize all possible gradient computation of (A.2), and arrive similar results as Section 3: almost linear algorithm for Problem 3. 

**Problem 3** (Approximate LoRA Gradient Computation (ALoRAGC( _L, d, r, ϵ_ ))) **.** Assume all numerical values be in log( _L_ ) bits encoding. Let _L_ follow (A.2), _ϵ >_ 0, and _∥Z∥∞_ := max _i,j |Zij|_ . The problem of approximating gradient computation of optimizing (A.2) is to find four surrogate gradient matrices _{G_<sup>�(</sup> _µ_<sup>_A_)</sup> _∈_ R<sup>_d×r_</sup> _, G_<sup>�(</sup> _µ_<sup>_B_)</sup> _∈_ R<sup>_r×d_</sup> _}µ_ = _K,Q_ such that 



**Theorem A.1** (Main Result: Existence of Almost Linear Time ALoRAGC) **.** Let Γ = _o_ (<sup>_√_</sup> log _L_ ). Suppose all numerical values are in O(log _L_ )-bits encoding. For _µ_ = _Q, K_ , let _Wµ_ = _Wµ_<sup>_⋆_+</sup><sup>_BµAµ∈_</sup> (1) (2) R<sup>_d×d_</sup> . If _Cµ_<sup>_W_</sup> _µ Cµ_ ��� ��� _∞_<sup>_≤_Γ and</sup> ��� ��� _∞_<sup>_≤_Γ for both</sup><sup>_µ_=</sup><sup>_Q, K_, then there exists a</sup><sup>_L_1+</sup><sup>_o_(1) time</sup> 

algorithm to solve ALoRAGC( _L, d_ = _O_ (log _L_ ) _, r_ = _L_<sup>_o_(1)</sup> _, ϵ_ = 1 _/_ poly( _L_ )) (i.e., Problem 3) up to 1 _/_ poly( _L_ ) accuracy. In particular, this algorithm outputs gradient matrices _{G_<sup>�(</sup> _µ_<sup>_A_)</sup> _∈_ R<sup>_d×r_</sup> _, G_<sup>�(</sup> _µ_<sup>_B_)</sup> _∈_ R<sup>_r×d_</sup> _}µ_ = _K,Q_ such that 



18 

Published as a conference paper at ICLR 2025 

_Proof._ See Appendix D for a detailed proof. 

19 

Published as a conference paper at ICLR 2025 

### B RELATED WORKS 

**Fine-Grained Complexity.** The Strong Exponential Time Hypothesis (SETH) is a conjecture in computational complexity theory that posits solving the Boolean satisfiability problem (SAT) for _n_ variables requires time 2<sup>_n_</sup> in the worst case, up to sub-exponential factors (Impagliazzo and Paturi, 2001). It extends the Exponential Time Hypothesis (ETH) by suggesting that no algorithm can solve _k_ -SAT in _O_ (2<sup>(1</sup><sup>_−ϵ_)</sup><sup>_n_</sup> ) time for any _ϵ >_ 0 (Calabro et al., 2009). SETH has significant implications for the hardness of various computational problems, as proving or disproving it would greatly enhance our understanding of computational limits (Williams, 2018b; 2013). In essence, SETH is a stronger form of the `P` = `NP` conjecture, suggesting that our current best `SAT` algorithms are optimal. It states as follows: 

**Hypothesis 2** (SETH) **.** For every _ϵ >_ 0, there is a positive integer _k ≥_ 3 such that _k_ - `SAT` on formulas with _n_ variables cannot be solved in _O_ (2<sup>(1</sup><sup>_−ϵ_)</sup><sup>_n_</sup> ) time, even by a randomized algorithm. 

SETH is widely used for establishing fine-grained lower bounds for various algorithmic challenges, including _k_ -Hitting Set and _k_ -NAE-SAT (Williams, 2018b; Cygan et al., 2016). This conjecture is crucial in deriving conditional lower bounds for many significant problems that otherwise have polynomial-time solutions in diverse fields such as pattern matching (Chen and Williams, 2019; Bringman and Künnemann, 2018; Bringmann et al., 2017; Bringmann and Mulzer, 2016; Backurs and Indyk, 2016; Bringmann, 2014; Abboud et al., 2014), graph theory (Dalirrooyfard et al., 2022; Chan et al., 2022; Abboud et al., 2018; Gao et al., 2018; Krauthgamer and Trabelsi, 2018; Roditty and Vassilevska Williams, 2013), and computational geometry (Karthik and Manurangsi, 2020; Williams, 2018a; Rubinstein, 2018; Chen, 2018; Buchin et al., 2016). 

Based on this conjecture, our study employs fine-grained reductions under SETH to explore the computational limits of Low-Rank Adaptation (LoRA). Previous research in fine-grained reductions includes the work by Backurs et al. (2017), who examine the computational complexity of various Empirical Risk Minimization problems, such as kernel SVMs and kernel ridge. Alman et al. (2020) investigate the effectiveness of spectral graph theory on geometric graphs within the constraints of SETH. Aggarwal and Alman (2022) address the computational limitations of Batch Gaussian Kernel Density Estimation. Expanding on these studies, Gu et al. (2024a;b); Alman and Song (2024b; 2023) explore transformer attention and introduced a tensor generalization. Alman and Yu (2024) establish the fundamental limitations on subquadratic alternatives to softmax transformers. Hu et al. (2024c) show that efficient dense associative memory a.k.a. modern Hopfield models and corresponding networks also need bounded query and key patterns for sub-quadratic time complexity. Compared to existing works, this work is, to the best of our knowledge, the first analysis of computational limits for parameter-efficient fine-tuning of large foundation models (Hu et al., 2021). 

**Low-Rank Adaptation (LoRA).** In this paper, we focus on LoRA (Hu et al., 2021), a method that leverages low-rank matrices to approximate updates to the weights of neural models. Various extensions of LoRA have been proposed to address different challenges in model training and deployment. For instance, DoRA (Liu et al., 2024) focus on enhanced parameter efficiency. QLoRA (Dettmers et al., 2024), LoftQ (Li et al., 2024), QA-LoRA (Xu et al., 2024b), and LQ-LoRA (Guo et al., 2024) focus on both memory and parameter efficiency in model compression and quantization. Additionally, DyLoRA (Li et al., 2020), AdaLoRA (Zhang et al., 2023), and SoRA (Ding et al., 2023) focus on dynamically determining the optimal rank _r_ for LoRA implementations. LoRAHub (Huang et al., 2023) focus on multi-task finetuning. LoRA+ (Hayou et al., 2024) focus on efficient feature learning. Despite the methodological and empirical successes, the theoretical side is relatively underdeveloped. While Zeng and Lee (2024) explore the expressiveness of LoRA from a universalapproximation perspective, and Hayou et al. (2024) investigate the optimal adapter learning rate with respect to large model width, to the best of our knowledge, no existing analysis focuses on the computational limits of LoRA. Therefore, this work provides a timely theoretical analysis of LoRA’s computational limits, aiming to advance efficient finetuning of large foundation models in terms of both parameter usage and computational time. 

**Outliers in Attention Heads.** Our results indicate that outliers (e.g., large _∥XW_<sup>_⋆_</sup> _∥_ and _∥XW_<sup>_⋆_</sup> + _αXBA/r∥_ ) in attention heads hamper LoRA efficiency and performance. This outlier effect is well-known in pretraining large foundation models for its negative impact on models’ quantization 

20 

Published as a conference paper at ICLR 2025 

performance (Sun et al., 2024). For pretraining, prior works identify the existence of no-op tokens as the main source: tokens with small value vectors tend to receive significantly large attention weights (Hu et al., 2024a; Bondarenko et al., 2023). Specifically, Hu et al. (2024a) interpret this outlier effect as inefficient _rare_ memory retrieval from the associative memory/modern Hopfield model perspective (Wu et al., 2024a;b; Xu et al., 2024a; Hu et al., 2025; 2024b;c; 2023) and propose the outlier-efficient Hopfield layer for transformer-based large models, demonstrating strong empirical performance and theoretical guarantees. The advantages of controlling outliers in the attention heads of transformer-based large foundation models are also emphasized in various theoretical studies (Gu et al., 2024a;b; Alman and Song, 2024a;b; 2023; Gao et al., 2023a). Yet, to the best of our knowledge, there is no existing work on outliers in LoRA fine-tuning. This is the first work establishing that the LoRA adaptor weights might lead to performance and efficiency degradation due to their additive nature: _∥XW_<sup>_⋆_</sup> + _αXBA/r∥_ . 

21 



<!-- Start of picture text -->
= Y y —(-—)<br>_t-. | ,<br>_ ({ —.—J ( )<br>—S—’<br>|<br>——= _ (-) ( )<br>( .) =<br>_ __ ( — ee ( _) )<br>——= f=.— ) ( )<br>(LL) | |<br>( -  _)<br>—(- —. )<br>—f 2 -y ¢ -yt EY<br>—<br>( ( -) ~~)<br><!-- End of picture text -->



<!-- Start of picture text -->
> r E -.( ( -) -.-)<br>~ e y -. { ( -) -.-)<br><!-- End of picture text -->



<!-- Start of picture text -->
Th e -. ( C -) ---)<br>> ( ( -) -.-_) --<br>( _ )<br>L e )<br>( () — )<br>— rye — ( ( -) —.-_)<br>= ( ( -) -.-) --<br>( _ )<br>yy )<br>() —  )<br>( ( ) ( ) )<br><!-- End of picture text -->



<!-- Start of picture text -->
| —|<br>| ee ee<br>( —) |<br>( )<br>()<br>“ —— ( )<br>on e<br>( ) ( )<br>~_ )(( ) )<br> - | ) |<br>| Tl — |<br>( )<br><!-- End of picture text -->



<!-- Start of picture text -->
| _|<br>| — _l )<br>| — -|<br>Ko) Ct ) 2 LI<br>~ ns oe )<br>( )<br><!-- End of picture text -->



<!-- Start of picture text -->
( -) ee ae<br>yo Fr)<br>o o )<br>I _|<br>ese --I<br>— ( fP-s-. -.-]) Fs. ---4 ( ] )<br><!-- End of picture text -->



<!-- Start of picture text -->
- ( € C ) -() )= =)<br>()<br>( )<br>()()<br>( )<br>( ) ~<br>()- -  ()<br>“Ty an<br>(() \C J<br>( ) ( )<br>( ) — ( Jo e<br>I -~ |<br>PC= )CCO ) J )<br>KC) - JCC I |<br>(o) - 7] ? (0) 8 i<br>a<br>( )<br>( )<br><!-- End of picture text -->



<!-- Start of picture text -->
> |<br>(=) (0) ~ F<br>( = ~ ~ J<br>(WM<br><!-- End of picture text -->



<!-- Start of picture text -->
|<br>— (J ~ (j)<br>a — (-) _<br>( __)<br>Sat y y y y _.<br>Sa) y y 2) r y<br><!-- End of picture text -->



~~<u>Oo</u>~~ 





~~Oo~~ 



<!-- Start of picture text -->
— Yer) ©) () ( ( ©)) ©) ©))<br>— YY) C) ( ) C € ©)) ©) ©))<br>TS E U) ) () ( ) ( ( ©)) ©) ©))<br>TS P ht) ) () ( ) ( ( ©)) ©) €))<br><!-- End of picture text -->



<!-- Start of picture text -->
— (() — )<br>— () —  )<br>— .) (() & ))<br>— «) () &) )<br>erce ) E C) (D E ( 0) ©) ©))<br>7 ( )<br>eo (0 ( 0)) ©) &)) &)<br>7 ( )<br>o r () ) . |<br>> (0) & )) Cy<br>e er) CV CI E ( 06) ©) ©)<br>Ee C Y C C C) ©) ©)) ©)<br>or () &)<br>> () ©) . )<br><!-- End of picture text -->

Published as a conference paper at ICLR 2025 

_GQ_<sup>(</sup><sup>_A_),</sup><sup>_G_(</sup> _Q_<sup>_B_),</sup><sup>_G_(</sup> _K_<sup>_A_),and</sup><sup>_G_(</sup> _K_<sup>_B_)),for</sup><sup>_µ_=</sup><sup>_Q, K_,we need to approximate the functions</sup><sup>_fµ_(</sup><sup>_·_),</sup><sup>_qµ_(</sup><sup>_·_),</sup> _cµ_ ( _·_ ), and thus _pµ_ ( _·_ ) with precision guarantees. To do so, it is convenient to consider the following decomposition of _pµ_ ( _·_ ) for _µ_ = _Q, K_ . 



**Overview of Our Proof Strategy.** Similar to Section 3, we adopt the following strategy: termby-term approximation for precision-guaranteed, almost linear time algorithms to compute LoRA gradients in Problem 3. For all _µ_ = _Q, K_ , we do the following. 

- **Step 1.** Prove the existence of almost linear approximation algorithms for _fµ_ ( _·_ ), _qµ_ ( _·_ ), and _cµ_ ( _·_ ) via low-rank approximation (Lemma D.5, Lemma D.7, and Lemma D.6). 

- **Step 2.** Prove the existence of almost linear approximation algorithms for _p_<sup>_µ_</sup> 1<sup>(</sup><sup>_·_),</sup><sup>_pµ_</sup> 2<sup>(</sup><sup>_·_),and thus</sup> _pµ_ ( _·_ ) via the low-rank-preserving property of the multiplication between _fµ_ ( _·_ ) and _qµ_ ( _·_ ) (Lemma D.8 and Lemma D.9). 

- **Step 3.** Prove the existence of almost linear approximation algorithms for the LoRA adapter gradients (i.e., _∂∂AL_ _~~Q~~_<sup>,</sup> _∂∂AL_ _~~K~~_<sup>,</sup> _∂∂BL_ _~~Q~~_<sup>, and</sup> _∂∂BL_ _~~K~~_<sup>in Lemma D.4) using the results from</sup><sup>**Step 1**and</sup><sup>**Step 2**</sup> (Theorem A.1). 

**Step 1.** We start with low-rank approximations for _fµ_ ( _·_ ) _, qµ_ ( _·_ ) _, cµ_ ( _·_ ). 

**Lemma D.5** (Approximate _fQ_ ( _·_ ) _, fK_ ( _·_ )) **.** Let Γ = _o_ (<sup>_√_</sup> log _L_ ), for _µ_ = _Q, K_ , suppose _Cµ_<sup>(1)</sup><sup>_, C_</sup> _µ_<sup>(2)</sup> _∈ ⊤_<sup>�</sup> R<sup>_L×d_</sup> , _W ∈_ R<sup>_d×d_</sup> , and _fµ_ <u>(</u> _<u>W</u>_ <u>)</u> = _D_<sup>_−_1</sup> exp _Cµ_<sup>(1)</sup><sup>_W_</sup> _Cµ_<sup>(2)</sup> with _D_ following (A.2). There � � � exists a _k_ 1 = _L_<sup>_o_(1)</sup> such that if _Cµ_ (1)<sup>_W_</sup> _Cµ_ (2) ��� ��� _∞_<sup>_≤_Γ and</sup> ��� ��� _∞_<sup>_≤_Γ, then there exist four matrices</sup> _U_ 1<sup>_Q, V_</sup> 1<sup>_Q, U_</sup> 1<sup>_K, V_</sup> 1<sup>_K_</sup> _∈_ R<sup>_L×k_1</sup> such that _U_ 1 _Q_<sup>(</sup><sup>_V_</sup> 1<sup>_Q_)</sup><sup>_⊤−fQ_</sup><sup><u>(</u></sup><sup>_<u>W</u>_</sup> <u>)</u> ��� ��� _∞_<sup>_≤ϵ/_poly(</sup><sup>_L_)</sup><sup>_,_</sup> �� _U_ 1 _K_<sup>(</sup><sup>_V_</sup> 1<sup>_K_)</sup><sup>_⊤−fK_</sup><sup><u>(</u></sup><sup>_<u>W</u>_</sup> <u>)��</u> _∞_<sup>_≤ϵ/_poly(</sup><sup>_L_)</sup><sup>_._</sup> In addition, it takes _L_<sup>1+</sup><sup>_o_(1)</sup> time to construct _U_ 1<sup>_Q, V_</sup> 1<sup>_Q, U_</sup> 1<sup>_K, V_</sup> 1<sup>_K_.</sup> 

_Proof._ This follows the proof of Lemma 3.3 



_Proof._ This follows the proof of Lemma 3.4 

**Lemma D.7** (Approximate _qQ_ ( _·_ ) _, qK_ ( _·_ )) **.** Let _k_ 2 = _L_<sup>_o_(1)</sup> , _cQ_ ( _W_ ) _, cK_ ( _W_ ) _∈_ R<sup>_L×d_</sup> follows Definition D.4 and let _qK_ <u>(</u> _<u>W</u>_ <u>)</u> := _C_<sup>(3)</sup> ( _cK_ <u>(</u> _<u>W</u>_ <u>))</u><sup>T</sup> _∈_ R<sup>_L×L_</sup> , _qQ_ <u>(</u> _<u>W</u>_ <u>)</u> := _C_<sup>(3)</sup> ( _cQ_ <u>(</u> _<u>W</u>_ <u>))</u><sup>T</sup> _∈_ R<sup>_L×L_</sup> . 

33 

Published as a conference paper at ICLR 2025 



_Proof._ This follows the proof of Lemma 3.5 

**Step 2.** Now, we use above lemmas to construct low-rank approximations for _p_<sup>_µ_</sup> 1<sup>(</sup><sup>_·_)</sup><sup>_, pµ_</sup> 2<sup>(</sup><sup>_·_)</sup><sup>_, pµ_(</sup><sup>_·_).</sup> **Lemma D.8** (Approximate _p_<sup>_Q_</sup> 1<sup>(</sup><sup>_·_)</sup><sup>_, p_</sup> 1<sup>_K_(</sup><sup>_·_))</sup><sup>**.**Let</sup><sup>_k_1</sup><sup>_, k_2</sup><sup>_, k_3=</sup><sup>_Lo_(1).</sup> For _µ_ = _K, Q_ , suppose _U_ 1<sup>_µ, V_</sup> 1<sup>_µ∈_R</sup><sup>_L×k_1approximate</sup><sup>_fµ_</sup><sup><u>(</u></sup><sup>_<u>W</u>_</sup> <u>)</u> _∈_ R<sup>_L×L_</sup> such that �� _U_ 1 _µ_<sup>(</sup><sup>_V_</sup> 1<sup>_µ_)</sup><sup>_⊤−fµ_</sup><sup><u>(</u></sup><sup>_<u>W</u>_</sup> <u>)��</u> _∞_<sup>_≤ϵ/_poly(</sup><sup>_L_),</sup> and _U_ 2<sup>_µ, V_</sup> 2<sup>_µ_</sup> _∈_ R<sup>_L×k_2</sup> approximate the _qµ_ <u>(</u> _<u>W</u>_ <u>)</u> _∈_ R<sup>_L×L_</sup> such that �� _U_ 2 _µ_<sup>(</sup><sup>_V_</sup> 2<sup>_µ_)</sup><sup>_⊤−qµ_</sup><sup><u>(</u></sup><sup>_<u>W</u>_</sup> <u>)��</u> _∞_<sup>_≤_</sup> _ϵ/_ poly( _L_ ). Then there exist two matrices _U_ 3<sup>_µ, V_</sup> 3<sup>_µ∈_R</sup><sup>_L×k_3such that</sup> �� _U_ 3 _µ_<sup>(</sup><sup>_V_</sup> 3<sup>_µ_)</sup><sup>_⊤−pµ_</sup> 1<sup><u>(</u></sup><sup>_<u>W</u>_</sup> <u>)��</u> _∞_<sup>_≤ϵ/_poly(</sup><sup>_L_)</sup><sup>_,_</sup> for _µ_ = _K, Q._ In addition, it takes _L_<sup>1+</sup><sup>_o_(1)</sup> time to construct _U_ 3<sup>_Q, V_</sup> 3<sup>_Q, U_</sup> 3<sup>_K, V_</sup> 3<sup>_K_.</sup> 

_Proof._ This follows the proof of Lemma 3.6 

**Lemma D.9** (Approximate _p_<sup>_Q_</sup> 2<sup>(</sup><sup>_·_)</sup><sup>_, p_</sup> 2<sup>_K_(</sup><sup>_·_))</sup><sup>**.**Let</sup><sup>_k_1</sup><sup>_, k_2</sup><sup>_, k_4=</sup><sup>_Lo_(1).Let</sup><sup>_pQ_</sup> 2<sup><u>(</u></sup><sup>_<u>W</u>_</sup> <u>)</u> _, p_<sup>_K_</sup> 2<sup><u>(</u></sup><sup>_<u>W</u>_</sup> <u>)</u> _∈_ R<sup>_L×L_</sup> such that its _<u>j</u>_ -th column is _p_ 2( _<u>W</u>_ <u>)</u> _<u>j</u>_ = _f_ <u>(</u> _<u>W</u>_ <u>)</u> _<u>jf</u>_ <u>(</u> _<u>W</u>_ <u>)</u><sup>_⊤_</sup> _<u>j</u> q_ <u>(</u> _<u>W</u>_ <u>)</u> _<u>j</u>_ follow Definition D.8, for each _<u>j</u> ∈_ [ _L_ ]. For _µ_ = _K, Q_ , suppose _U_ 1<sup>_µ, V_</sup> 1<sup>_µ_</sup> _∈_ R<sup>_L×k_1</sup> approximates the _fµ_ <u>(</u> _<u>W</u>_ <u>)</u> such that <u>��</u> _U_ 1 _µ_<sup>(</sup><sup>_V_</sup> 1<sup>_µ_)</sup><sup>_⊤−fµ_</sup><sup><u>(</u></sup><sup>_<u>W</u>_</sup> <u>)��</u> _∞_<sup>_≤ϵ/_poly(</sup><sup>_L_), and</sup><sup>_U_</sup> 2<sup>_µ, V_</sup> 2<sup>_µ∈_R</sup><sup>_L×k_2approximates the</sup><sup>_qµ_</sup><sup><u>(</u></sup><sup>_<u>W</u>_</sup> <u>)</u> _∈_ R<sup>_L×L_</sup> such that �� _U_ 2 _µ_<sup>(</sup><sup>_V_</sup> 2<sup>_µ_)</sup><sup>_⊤−qµ_</sup><sup><u>(</u></sup><sup>_<u>W</u>_</sup> <u>)��</u> _∞_<sup>_≤ϵ/_poly(</sup><sup>_L_).Then there exist matrices</sup><sup>_U_</sup> 4<sup>_µ, V_</sup> 4<sup>_µ∈_R</sup><sup>_L×k_4such</sup> that 

�� _U_ 4 _µ_<sup>(</sup><sup>_V_</sup> 4<sup>_µ_)</sup><sup>_⊤−pµ_</sup> 2<sup><u>(</u></sup><sup>_<u>W</u>_</sup> <u>)��</u> _∞_<sup>_≤ϵ/_poly(</sup><sup>_L_)</sup><sup>_,_</sup> for _µ_ = _K, Q._ In addition, it takes _L_<sup>1+</sup><sup>_o_(1)</sup> time to construct _U_ 4<sup>_Q, V_</sup> 4<sup>_Q, U_</sup> 4<sup>_K, V_</sup> 4<sup>_K_.</sup> 

_Proof._ This follows the proof of Lemma 3.7 

**Step 3.** Combining above, we arrive our main result: almost linear algorithm for Problem 3. 



_Proof of Theorem A.1._ By the definitions of matrices _p_<sup>_K_</sup> 1<sup><u>(</u></sup><sup>_<u>W</u>_</sup> <u>)</u> _, p_<sup>_Q_</sup> 1<sup><u>(</u></sup><sup>_<u>W</u>_</sup> <u>)</u> _, p_<sup>_K_</sup> 2<sup><u>(</u></sup><sup>_<u>W</u>_</sup> <u>)</u> _, p_<sup>_Q_</sup> 2<sup><u>(</u></sup><sup>_<u>W</u>_</sup> <u>) in Defini-</u> tion D.8 and _pK_ <u>(</u> _<u>W</u>_ <u>)</u> _, pQ_ <u>(</u> _<u>W</u>_ <u>) in Definition D.7.</u> It is straightforward that 



34 



<!-- Start of picture text -->
— (() © ))<br>— () G) )<br>— () ((€) & ))<br>— ( ) ( ) EC ) )<br>—_- — ( (  ))<br>C0) ©) ) () ©) )<br>() (() CC ) )<br>() ( ) CC ) )<br>() (() CC ) )<br>()<br>() ( ) C&C ) )<br>() ( ) C&C ) )<br>I-- Jo<br>hoo (6) © ))6) 6 0 )6) DI<br>he on 6) ©) ) 66 )- e) )<br>MC )C C )- C) ICCC ) e y dD<br>hid de ©) > Cyr W e) > eI)<br><!-- End of picture text -->



<!-- Start of picture text -->
I |<br>ho 6 () ©) ) od ED ) DI<br>kod Ie) OY ) ) ED I<br>M CE) YE WCE ) e y ) DI<br>hid Pde Ge ) > ey) I e) > eI)<br><!-- End of picture text -->

Published as a conference paper at ICLR 2025 

### E PROOF OF THEOREM 4.1 

We recall our definition of ALoRAGC( _L, d, r, ϵ_ ) for special case from Problem 2 subject to LoRA loss (3.3). We aim to make the reduction from AAttLGC( _L, r, ϵ_ ) (Alman and Song, 2024a, Definition 1.4) to our problem ALoRAGC( _L, d, r, ϵ_ ). 

**Definition E.1** (Approximate Attention Loss Gradient Computation (AAttLGC( _L, r, ϵ_ )), Definition 1.4 of (Alman and Song, 2024a)) **.** Given four _L × r_ size matrices _A_ 1 _∈_ R<sup>_L×r_</sup> _, A_ 2 _∈_ R<sup>_L×r_</sup> _, A_ 3 _∈_ R<sup>_L×r_</sup> , _E ∈_ R<sup>_L×r_</sup> and a square matrix _X ∈_ R<sup>_r×r_</sup> to be fixed matrices. Assume that _∥A_ 1 _X∥∞ ≤ B_ , _∥A_ 2 _∥∞ ≤ B_ . Assume all numerical values are in log( _L_ )-bits encoding. Let _L_ ( _X_ ) :=<sup><u>1</u></sup> 2<sup>_∥D−_1 exp</sup> � _A_ 1 _XA_<sup>_⊤_</sup> 2<sup>_/r_</sup> � _A_ 3 _− E∥_<sup>2</sup> _F_<sup>_._which</sup><sup>_D_:= diag(exp</sup> � _A_ 1 _XA_<sup>_⊤_</sup> 2<sup>_/r_</sup> �1 _L_ ) _._ Let<sup>d</sup><sup>_L_</sup> d<sup><u>(</u></sup> _X_<sup>_X_</sup><sup><u>)</u></sup> denote the gradient of loss function _L_ . The goal is to output a matrix � _g ∈_ R<sup>_L×L_</sup> such that 



We recall the main hardness result of (Alman and Song, 2024a) which shows a lower bound of AAttLGC( _L, r, ϵ_ ) (Definition E.1) in the following particular case by assuming SETH. 

**Lemma E.1** (Theorem 5.5 of (Alman and Song, 2024a)) **.** Let _κ_ : N _→_ N by any function with _κ_ ( _L_ ) = _ω_ (1) and _κ_ ( _L_ ) = _o_ (log _L_ ). Assuming SETH, there is no algorithm running in time _O_ ( _L_<sup>2</sup><sup>_−δ_</sup> ) for any constant _δ >_ 0 for Approximate Attention Loss Gradient Computation AAttLGC( _L, r, ϵ_ ), even in the case where _r_ = _O_ (log _L_ ) and the input matrices satisfy _∥A_ 1 _∥∞, ∥A_ 2 _∥∞, ∥A_ 3 _∥∞ ≤ O_ (<sup>_√_</sup> log _L · κ_ ( _L_ )) = _B_ , _E_ = 0, _X_ = _λIr_ for some scalar _λ ∈_ [0 _,_ 1], and _ε_ = _O_ (1 _/_ (log _L_ )<sup>4</sup> ). 

Finally, we are ready for our main proof of Theorem 4.1. 

_Proof._ Considering Problem 2, we start with the following _O_ (1) reduction. Given the instance of AAttLGC( _L, r, ϵ_ ) and _A_ 1 _∈_ R<sup>_L×r_</sup> , _A_ 2 _∈_ R<sup>_L×r_</sup> , _A_ 3 _∈_ R<sup>_L×r_</sup> , _E_ = 0, _B_ = _O_ (<sup>_√_</sup> log _L · κ_ ( _L_ )). We then transfer this instance to the instance of ALoRAGC( _L, d, r, ϵ_ ) by making the following substitution: 



Then we have _∥C_<sup>(2)</sup> _∥∞, ∥C_<sup>(1)</sup> _BQAQ∥∞, ∥Y ∥∞ ≤_ Γ such that 



and hence 



This implies that the upper _L × r_ subblock is exactly the same. (Here we can assume _E_ = _Y_ = 0.) 



This follows that the derivative with respect to _X_ of the RHS is the same as the partial derivative � with respect to _AQ_ by embedding _X_ into a subblock of _AQ_ . Now, by letting _G_<sup>�</sup> _A_ = _g_ in the AAttLGCC( _L, r, ϵ_ ), which finishes the reduction. This completes the proof. 

37 

Published as a conference paper at ICLR 2025 

F QUADRATIC TIME COMPLEXITY OF EXACT LORA GRADIENT COMPUTATION 

Here, we make more comments on tensor-trick decomposed LoRA loss from Lemma 3.1: 



**Remark F.1** (Benefit from Tensor Trick: Speedup Seemingly Cubic Time Exact Computation) **.** Lemma 3.1 highlights the benefits of the tensor trick and the potential for speeding up _exact_ LoRA adaptation on transformer-based models. To be more specific, for any _<u>j</u> ∈_ [ _L_ ], **Part-(I)** is an _L × L_ matrix, thus requiring Θ( _L_<sup>2</sup> ) time to compute. Moreover, with a total of _L_ terms, the overall computation time amounts to Θ( _L_<sup>3</sup> ). 

However, (3.5) decomposes **Part-(I)** into a _diagonal_ **Part-(II)** and a _low-rank_ **Part-(III)** (specifically, rank-1). This decomposition allows us to reduce the computation time of **Part-(I)** to _O_ ( _L_ ) for each _<u>j</u> ∈_ [ _L_ ], and of the entire<sup>d</sup><sup>_L_(</sup><sup>_<u>W</u>_</sup> ) _/_ <u>d</u> _<u>W</u>_ to _O_ ( _L_<sup>2</sup> ). Our next theorem verifies this claim and shows such seemingly cubic time exact computation is in fact quadratic. 

**Definition F.1.** Let _n_ 1 _, n_ 2 _, n_ 3 denote any three positive integers. We use _T_ mat( _n_ 1 _, n_ 2 _, n_ 3) to denote the time of multiplying an _n_ 1 _× n_ 2 matrix with another _n_ 2 _× n_ 3. 

**Theorem F.1** (Exact LoRA Gradient Computation Takes Quadratic Time) **.** Suppose the following objects are given and if following conditions hold, 

• Let _C_<sup>(1)</sup> _, C_<sup>(2)</sup> _, C_<sup>(3)</sup> _∈_ R<sup>_L×d_</sup> be in (3.2). Let _BQ ∈_ R<sup>_d×r_</sup> _, AQ ∈_ R<sup>_r×d_</sup> _, W ∈_ R<sup>_d×d_</sup> be in (3.3). 

• Let _f_ ( _·_ ) _, c_ ( _·_ ) _, p_ 1( _·_ ) _, p_ 2( _·_ ) follow from their definitions in Section 3. 

• Let _<u>G</u>_ ( _A_ ) := _<u>∂L</u>_<sup>_<u>G</u>_</sup> ( _B_ ) := _<u>∂L</u>_ _~~Q~~ ∂AQ_<sup>_,_</sup> _~~Q~~ ∂BQ_<sup>(Where</sup><sup>_L_is defined in (3.3) ).</sup> ( _A_ ) ( _B_ ) Then we can make _exact_ computation of _<u>G</u>_ _~~Q~~_<sup>_, G_</sup> _~~Q~~_ in _O_ ( _T_ mat( _d, L, L_ ) + _T_ mat( _d, d, L_ ) + _T_ mat( _d, d, r_ )) time. 

_Proof._ Due to Lemma 3.2, it holds 



Recall that the decomposition of _p_ <u>(</u> _<u>W</u>_ <u>)</u> = _p_ 1( _<u>W</u>_ <u>)</u> _− p_ 2( _<u>W</u>_ <u>).</u> And according to Definition 3.6, for every index _<u>j</u> ∈_ [ _L_ ], 



In addition, due to Lemma 3.2, _q_ <u>(</u> _<u>W</u>_ <u>) is defined as</u> 



Therefore, we compute _f_ <u>(</u> _<u>W</u>_ <u>)</u> _, c_ <u>(</u> _<u>W</u>_ <u>)</u> _, p_ 1( _<u>W</u>_ <u>)</u> _, p_ 2( _<u>W</u>_ <u>) in order as follows.</u> Then we combine them together to get total running time. 

- **Step 1.** We compute _f_ <u>(</u> _<u>W</u>_ <u>).</u> 

Note that 



where 



38 

Published as a conference paper at ICLR 2025 

We firstly compute exp� _C_<sup>(1)</sup> _W_ ( _C_<sup>(2)</sup> )<sup>_⊤_�</sup> _C_<sup>(3)</sup> which takes time of _T_ mat( _d, d, L_ ) + _T_ mat( _d, L, L_ ). 

Then, we can compute _D_ which takes _O_ ( _L_<sup>2</sup> ) time. 

Then, we can compute _f_ <u>(</u> _<u>W</u>_ <u>) which takes</u> _O_ ( _L_<sup>2</sup> ) time. 

Thus, the overall time is 



Therefore, the proof is completed. 

- **Step 2.** We compute _c_ <u>(</u> _<u>W</u>_ <u>).</u> Based on the Definition 3.5, which is 



Computing _f_ <u>(</u> _<u>W</u>_ <u>)</u> _C_<sup>(3)</sup> takes time of _T_ mat( _d, L, L_ ) and computing _f_ <u>(</u> _<u>W</u>_ <u>)</u> _C_<sup>(3)</sup> _− Y_ takes time of _O_ ( _Ld_ ). Thus, the overall time is _T_ mat( _d, L, L_ ) + _O_ ( _Ld_ ) = _O_ ( _T_ mat( _d, L, L_ )). 

- **Step 3.** We compute _q_ <u>(</u> _<u>W</u>_ <u>).</u> Recall that 



Therefore, it takes time _O_ ( _T_ mat( _d, L, L_ )). 

- **Step 4.** We compute _p_ <u>(</u> _<u>W</u>_ <u>).</u> Note that due to Definition 3.6, which is 



such that _p_ <u>(</u> _<u>W</u>_ <u>) =</u> _p_ 1( _<u>W</u>_ <u>)</u> _− p_ 2( _<u>W</u>_ <u>).</u> 

Since diag( _f_ <u>(</u> _<u>W</u>_ <u>)</u> _<u>j</u>_ <u>) is a diagonal matrix and</u> _f_ <u>(</u> _<u>W</u>_ <u>)</u> _<u>j</u>_ <u>(</u> _f_ <u>(</u> _<u>W</u>_ <u>)</u> _<u>j</u>_ <u>)</u><sup>_⊤_</sup> is a rank-one matrix, we know that _p_ <u>(</u> _<u>W</u>_ <u>)</u> _<u>j</u> ∈_ R<sup>_L_</sup> can be computed in _O_ ( _L_ ), for each _<u>j</u> ∈_ [ _L_ ]. Thus we can construct matrix _p_ <u>(</u> _<u>W</u>_ <u>)</u> _∈_ R<sup>_L×L_</sup> in _L × O_ ( _L_ ) = _O_ ( _L_<sup>2</sup> ) time in total. 

- **Step 5.** Using Lemma 3.2, we know that 



Suppose _BQ ∈_ R<sup>_d×r_</sup> _, AQ ∈_ R<sup>_r×d_</sup> _, C_<sup>(1)</sup> _, C_<sup>(2)</sup> _, C_<sup>(3)</sup> _∈_ R<sup>_L×d_</sup> are given, then each of the gradients can be computed in time of _O_ ( _T_ mat( _d, L, L_ ) + _T_ mat( _d, d, L_ ) + _T_ mat( _d, d, r_ )). 

Thus, the overall running time for gradients computation is 



This completes the proof. 

39 

~~a ~ ~~~ 



<!-- Start of picture text -->
Mmm LoRA on Standard Transformer<br>@@m™m LoRA on Outlier-Free Transformer<br>75<br>vo<br>3 50<br>£<br>S 25<br>0 OPT-125m OPT-350m OPT-1.3b<br><!-- End of picture text -->


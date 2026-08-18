Natural Language Processing 

Final Project: Instructions 

**Natural Language Processing** 

### **Tel Aviv University** 

# **Final Project: Instructions** 

Lecturer: Dr. Mor Geva 

### **Important dates** 

Project proposal submission deadline: **July 7, 2026** Project submission deadline: **September 30, 2026** 

# **1 Goals** 

The course project provides an exciting opportunity to: 

- Conduct independent research in NLP. You will choose a paper (or a couple of papers) of your choice and focus on some of its aspects: either a possible extension, a related or modified model, a novel evaluation method or analysis, etc. Alternatively, you can choose a topic you are interested in and expand the scientific knowledge about it. 

- Potentially contribute to science. While it would be great if your project yields new results, findings, or resources (in which case we will consider pursuing publication in an upcoming NLP/AI conference), to get full credit it is sufficient to describe what you wanted to achieve, how you tried to achieve it, and what were the challenges you faced, as well as how it fits into the existing knowledge around this topic. 

- Experience research and the joy of new discoveries! 

# **2 Project Instructions** 

## **2.1 Proposal submission** 

- (a) Choose your partner(s): work in groups of 2-4 students. Working in teams may involve some logistical efforts, but it is often worth the tradeoff, as more teammates means more working hands and more heads to brainstorm with. 

- (b) Choose one to three (but not more than three!) papers that are related to the course topics and were published in NLP/ML conferences (e.g., EMNLP, *ACL, ICLR, NeurIPS) ideally after 2018. Other papers/venues can be considered but need my approval. 

- (c) Submit a proposal as described below for your project by **July 7, 2026** and get a binary approval from me. **Submitting a project proposal is mandatory** and subject to the same late-penalties as the homework assignments (affecting your final project grading). 

**Proposal** The project proposal should be up to a single page, and include the following information: 

1. Project title 

Page 1 of 9 

Natural Language Processing 

Final Project: Instructions 

2. Names, IDs, and emails of the contributing students. 

3. A brief description of the proposed project: what problem or research question you will tackle, what is the motivation for studying this problem/question (why is it interesting?), what is already known and what you are hoping to achieve. It should be clear from reading the proposal what you intend to do and roughly how, and how you will evaluate your success. The main goal for me is to understand that the proposed project is sufficient, but also not overly-ambitious. 

4. Make sure to describe your assumptions and requirements for the project (e.g., access to compute, access to API resources, access to specific data, human evaluators, model weights, etc.) and how you plan to attain them. 

5. During the project, you will have access to the GPU cluster of Tel-Aviv Data Center. If you have specific needs that I may be able to accommodate (such as access to cloud API endpoints), please include them as well and we will see if it is possible. 

6. If you work on a mentored project (see §5), you can simply submit that you will work on that, or extend the description if you have any additional plans around it. 

## **2.2 Computational resources** 

All students will have access to the Slurm cluster, using the partition `studentkillable` (see usage instructions in `https://www.cs.tau.ac.il/system/slurm` ), and could run jobs on the GPUs there. **Before you start, make sure to fill in this form:** `https://www.cs.tau.ac.il/system/SlurmRequestForm` (in the PI/lab write “NLP class 2025/2026”). 

In addition, you will have access to dedicated storage where you can host your models and datasets. This storage is persistent (though not backed-up, so make sure you back up all of your code and intermediate critical results on GitHub/Google drive) and will be available for you under 

`/home/morg/NLP` ~~`2`~~ `526b/<your` ~~`u`~~ `ser name>` anywhere on the system machines, starting from the end of the semester and **until the project submission due date** . Make sure you use it for data, cache, virtual env, default directory for HuggingFace etc. See my tips in the wrap-up slides and check out the tutorials I linked to. You can also consider linking e.g. _∼_ `/.cache` to a directory there with `ln -s /home/morg/NLP` ~~`2`~~ `526b/<your` ~~`u`~~ `ser name>/.cache` _∼_ `/.cache‘` . **Please avoid overusing it** . For example, if you fine-tune a model, make sure you don’t save a new checkpoint every 500 steps as it will quickly block the storage space. 

In addition to the resources above, you may find that you need some special resources, such as OpenAI/Gemini API credits or GPUs beyond what is available to you via Slurm. If you do, please email me ASAP, explaining your needs with the following details: 

1. Group names, IDs and emails 

2. Project title 

3. As exact as possible estimates of your needs. For example, if you need API access, include how much credits you need and to which models (you can use OpenAI online tokenizer and pricing to compute that). If you need GPU access, to which GPUs and how many hours to think you will need. Same goes for extra storage, etc. 

4. A short textual justification for this need, for me to pass on for approval. 

Page 2 of 9 

Natural Language Processing 

Final Project: Instructions 

**These are not guaranteed to be fulfilled, but I will do my best.** Always have a plan B and design your research plan so that you do most of the work on the resources you already have, saving the limited resources to the final experiments as much as possible. 

## **2.3 Final submission** 

The final project will be handed in as a “paper” in ACL format; see: 

https://www.overleaf.com/latex/templates/association-for-computational-linguistics-acl-conference/jvxskxpnznfj 

The report length is limited to 8 pages (not including references and appendix). It can be shorter, but make sure you cover all the required parts (see §3). If needed, after submission, I will schedule a short meeting with some groups to discuss their report. 

# **3 Grading** 

Projects will be graded according to the guidelines below. While not all points are applicable to all projects, please read them carefully to get a sense of what should be considered when writing your project report. 

**Quality of research question (10pt)** The project should start with a clear and concise research question that is relevant to the subject of the class. The question should be original and significant. Choosing a research question is part of the project proposal phase. 

**Ambitiousness and effort (10pt)** This considers the complexity and novelty of the chosen research question, the innovation in methodology or approach, the potential impact of the research, and the demonstrated effort in pursuing an ambitious and challenging project. This criterion rewards students for intellectual risk-taking and innovation, and for pushing the boundaries of their knowledge and skills, even if the final results may not be as polished or conclusive as less ambitious projects. 

**Literature review (20pt)** The project should include a well-written comprehensive literature review, providing background on prior research on the topic and contextualizing your work within it. It should critically evaluate previous research and identify gaps in the literature that the project addressed. This criterion also considers proper citation of sources used in the project. When mentioning results or claims from prior work, you should give the correct attribution. The citation style should be consistent and the bibliography should be formatted correctly (if you have not used the _\_ `citet` and _\_ `citep` macros, you probably used the wrong format). 

**Methodology (20pt)** Assess the methodology used to answer the research question. The methods should be appropriate for the question and be rigorously executed. Depending on the nature of the project, this might involve an evaluation of experimental design, data collection and analysis, or theoretical argumentation. This clause also asserts that your methodology is aligned with the project’s expected scope. Points for consideration: 

Page 3 of 9 

Natural Language Processing 

Final Project: Instructions 

- When performing experiments, always make sure you follow best-practices such as avoiding data leakage and overfitting (remember train/validation/test splits?), using appropriate models and sensible hyperparameters. Also, always compare with relevant baselines. For example, a naive majority vote classifier and/or the most common approach that is currently used in the literature. This is the only way to frame your results in any meaningful way. Similarly, always account for randomness (e.g. seeds, prompts affect, randomness from the decoding scheme). 

- If you train a model and can’t get any meaningful results, at least make sure you are able to overfit on a small sample of the data – otherwise you are doing something wrong and probably have a bug. Show me that the sanity experiment worked. 

- Make sure that your results are reproducible (as much as possible). This is usually done by specifying the exact settings in which you ran the experiment and providing access to the code and data used. When not possible, clearly state the reason. 

**Results and discussion (20pt)** The paper must present the findings accurately and coherently. The discussion should contextualize these results within the research question and existing literature. Both positive and negative results are valued, provided they stem from sound methodology, are thoroughly analyzed, and meaningfully contribute to the discourse. Points for consideration: 

- When you present results, you should discuss what conclusions are derived from them. 

- Always make sure your results are given in full and in an easy-to-understand format (whether it is a table, a scatter plot, a bar plot or simply inline text). 

- Dataset statistics: when working with datasets, it is often helpful for the reader to understand how the data was built, its size, and if annotated, what is the label distribution. When the samples’ description is not trivial, also consider including examples. If you created data, make sure to describe the process through which you created it. 

- Figures: it is often useful to have a figure/algorithm box outlining your method (when appropriate) which both helps the reader understand what you are describing and lets you refer to it while outlining your method. Additionally, it is often a nice touch to add figure in the first or second page with some demonstration of your main results, to which you can refer from your introduction when discussing your contributions. 

**Presentation and communication (20pt)** This assesses the clarity of writing, narrative coherence, grammatical accuracy, and visual clarity of figures and explanations. It evaluates the overall organization and aesthetic presentation of the work, highlighting the importance of conveying complex research in a clear, concise, and engaging manner. This is also where you are evaluated on your division of content to the proper sections and following the expected format (ACL template). Points for consideration: 

- Structure: The introduction should set the stage to what problem you are tackling, what high-level ideas you used to solve the problems, and what your main findings are. A thorough background is usually best placed in its own section. 

- Make sure the paper is self-contained. Namely, ensure that any uncommon models and metrics are clearly defined, and all the important background is there. If you rely on previous work, make sure it is accessible and properly cited. 

Page 4 of 9 

Natural Language Processing 

Final Project: Instructions 

- When using LLMs, make sure you stand behind what it writes. Beware of using it too much. While on the surface the generated content sounds good, very often the text becomes too boastful and not succinct, hiding the important details in too much text. 

- Avoid compiling a detailed work log with every step and issue encountered. Instead, focus on creating a white paper. For example, rather than ’We tried X but it failed because of Y, leading us to try Z’, write ’X was ineffective due to Y. Z proved successful. Implementation details are in...’. Be concise and omit exhaustive challenges. 

- While the appendix is a great place to put additional figures, proofs or examples, it should not contain any content that is critical to the flow of the paper. In fact, it should be assumed that the appendix is not read at all unless the reader is specifically interested in certain details. 

- Figures should use large enough fonts and added as pdf files (not jpeg or png) to ensure high quality. Feel free to use LLMs to help you create beautiful plots easily! 

# **4 Policy on AI Usage** 

You are welcome to use AI tools in your project, under the following guiding principles: 

- **Ownership** : Use AI tools to support your thinking, learning, coding and writing – not replace them (for motivation see [5, 6]). Maintain ownership of your work; your submission in its totality should reflect your own intellectual work and understanding, not the AI’s. 

- **Accountability** : You have full responsibility for the content in your final project report, including any potential mistakes. 

- **Transparency** : You are asked to include a brief “AI Disclosure and Reflection” section, detailing your usage in AI for the project. Explain how, why, in which parts, and for what purpose you used AI. Mention which AI tools and models you used, and reflect on your experience (i.e., how did it go?). This _will not_ impact the evaluation of your submission – we are aiming for transparency, not policing. We request this to provide you with habits of responsible AI usage and for us to gather insights and improve our teaching. 

# **5 Project ideas** 

There are many interesting questions to answer with the tools you have learned in class. Try to avoid ideas that require significant engineering (e.g., building an online platform), significant human efforts (e.g., large scale annotations) or significant compute (e.g., reproducing GPT-4). Instead, you can think of ways to improve the evaluation of existing methods, identifying problems that LLMs struggle with, coming up with new interesting tasks (and working on a proof-of-concept for them), improving a model for a specific task or extending our knowledge into tools that are widely used. 

While the grading guideline section might sound daunting, the scope of the project is a white-paper with a proof-of-concept presentation of an idea, and not a scientific paper ready to be submitted to an international conference. Choosing a good idea is key to make sure your research and implementation phase is both interesting and can be completed in a few days of work, leaving you with meaningful results to be compiled into a white paper. 

Page 5 of 9 

Natural Language Processing 

Final Project: Instructions 

## **5.1 Default project scope** 

Several model suites, such as OLMo and LMEnt [3], include pre-trained/fine-tuned models and their training data, often with sequence-level metadata information (i.e. which sequences the model saw during each step of training). These suites provide an easy-to-use framework for studying various problems in LLMs and NLP. You can think about it as a playground where we can train, intervene, and analyze small-scale language models with full visibility into their training data. Keep in mind that these models are not always as performant as leading LLMs. Still, as leading LLMs rely on similar architectures and their pre-training process is (likely) similar, conclusions from these small models could potentially transfer to them. Interesting problems one could explore include: 

- Understanding training dynamics – how and when different capabilities are obtained by models. 

- Knowledge and hallucinations – how and where in the model knowledge is being encoded during training, when do models start hallucinating, and more. 

- Better selection of training data instances – we may be able to make training more efficient by removing data points that do not contribute to improving the model’s performance. 

- Memorization and generalization 

- Adversarial settings (data privacy, malicious content injection into training data, biases, etc) + data contamination 

- Exploring better architectures (mostly for knowledge-related capabilities) 

- Many more, be creative! 

Some of these problems could also be explored as part of the BabyLM challenge, where the training data is fixed and designed for training a small LM. 

Before choosing a project idea, please make sure you have a good understating of how you will approach the subject, and that you have access to any required resources to complete what you plan. 

## **5.2 Mentored projects** 

The projects listed below were suggested by NLP researchers/research students and can be pursued under their guidance, potentially with the goal of achieving a publication. These projects are more extensive and may demand substantial effort and dedication, but they also offer the advantages of collaborating with experienced researchers. While you are free to work on these ideas independently, mentorship is contingent upon their consent. **If you are interested in pursuing any of these ideas, please email me, and I will connect you with the mentor for an initial meeting to assess compatibility. MFA-Based Steering via Membership Editing (mentored by PhD student Or Shafran)** _Background:_ Steering is a technique that alters an LM’s behavior, tone, or line of thought without needing to retrain or fine-tune the underlying model. It works by directly intervening in the model’s internal processing (via its vectors or math) to nudge it toward or away from specific concepts. Standard steering methods often edit activations using a global direction. This can change the target concept, but it may also damage other information encoded in the same activation, such as topic, entity identity, syntax, or factual context. This project tests whether MFA [8] (a new decomposition method) can make affine steering [9] more selective. Affine steering views control as moving activations from a source-concept 

Page 6 of 9 

Natural Language Processing 

Final Project: Instructions 

distribution to a target-concept distribution, for example, by matching their means or covariances. However, in high-dimensional activation space, it is not obvious how to define the relevant source and target distributions for a specific activation. MFA, or Mixture of Factor Analyzers, models activation space as a mixture of local Gaussian regions. Each region has a centroid and a low-dimensional subspace that captures variation inside the region. This gives a practical way to describe an activation by both its region membership and its local coordinates. We can therefore apply affine steering locally: move the activation toward target-concept regions, while preserving its within-region variation. Concretely, given an activation, we infer its MFA component membership and local coordinates. To steer a concept, we move it toward a target-concept centroid or modify its membership weights, while keeping the local coordinates fixed. For a stronger version, we can also use covariance matching between source and target MFA regions. 

_Hypothesis:_ Concept changes can often be achieved by changing MFA membership without destroying prompt-specific information carried by the local variation. 

_Evaluation:_ Compare against global steering, measure concept success, fluency, and preservation of nontarget attributes. Can use AxBench [11] for a proper evaluation. 

_Expected outcome:_ a more controlled steering method that changes high-level concept membership while preserving local details, showing that MFA can separate “which semantic region the activation belongs to” from “what information it carries inside that region.” 

**Robust Concept Erasure via Precise Editing of Token Embeddings (mentored by MSc student Clara Suslik)** _Concept erasure_ aims to persistently remove a target concept (e.g. _Harry Potter_ ) from a language model’s parameters, so the model can no longer reliably produce knowledge about it while retaining unrelated capabilities. Most methods edit the MLP layers, yet the erased knowledge often remains recoverable. Our recent work [4], shows that part of this stems from overlooking the _embedding_ layer: EMBER uses Sparse Matrix Factorization to decompose token embeddings into interpretable features, identifies those tied to the target concept, and subtracts them from a small set of concept-related tokens. As a plug-n-play module ensembled with existing erasure methods, it improves erasure precision and substantially increases robustness. 

The list below offers three independent project options, each extends this line of work. Each gives a direction, not a fixed protocol; the research question and experimental design are yours to develop. The code and data are available to build on. 

- **Project 1: Cross-lingual erasure and multilingual editing.** Because EMBER edits English tokens, erasure may not transfer when the same concept is accessed through another language. This project has two parts. First, measure cross-lingual leakage: translate the evaluation questions into one or two languages and compare how much concept knowledge each method recovers, and whether embedding-based erasure degrades differently from MLP-based methods. Second, extend the editing pipeline to multilingual data and study whether the results reproduce in other languages, whether the recovered features are shared across languages or language-specific, and how differences in tokenization and per-language token counts affect the method. 

- **Project 2: Comparing erasure against a model that never learned the concept.** Erasure is normally evaluated against the original model, but the natural gold standard is a model that simply never saw the concept in training. Recent data-controlled training suites make this comparison feasible. We suggest starting from **LMEnt** [3], whose entity-annotated Wikipedia corpus and checkpoints let 

Page 7 of 9 

Natural Language Processing 

Final Project: Instructions 

one cleanly include or exclude a concept’s data and retrain; related suites such as **Hubble** [10] (paired standard/perturbed models with controlled text insertion) and the **Deep Ignorance** [7] models (pretrained on data filtered to remove a target topic) offer complementary setups. The project trains matched models with and without the target concept’s data, then asks how close existing erasure methods get to this never-learned baseline: behaviorally, distributionally (e.g. KL divergence between the two models’ outputs), and under robustness evaluations. 

**Generalizability of Unlearning Fingerprints Across Varied Knowledge Targets (mentored by MSc student Shir Rashkovits)** Large Language Model (LLM) unlearning has emerged as a key alignment strategy to remove copyrighted, harmful, or private data. However, recent work by [1] demonstrates that machine unlearning is not invisible; it leaves systematic “fingerprints” (unlearning traces) in both model outputs and internal token logits, allowing simple classifiers to detect whether a model has undergone unlearning with over 90% accuracy. While prior work successfully detected these traces, it heavily relied on evaluations where the classifier was tested on the same or highly similar knowledge domains as the unlearning target. This project aims to expand this critical paradigm by investigating a fundamental research question: _Are unlearning fingerprints specific to the deleted knowledge content, or are they an artifact of the unlearning methodology itself?_ 

We propose a comprehensive Cross-Target Evaluation Framework. We will train probe classifiers to detect hidden unlearning traces on a specific target domain and then evaluate their generalization capabilities on entirely disjoint, unseen knowledge domains that were unlearned using the same method. To achieve this, the project will establish a diverse set of completely uncorrelated unlearning targets, such as Cyber-security weapons, copyrighted fiction, and Biosecurity hazards. If a classifier trained solely on “Cyber-weapons removal traces” can accurately detect when a model has undergone “copyrighted fiction removal,” it provides definitive empirical proof that unlearning algorithms introduce global, structural distortions to the model’s internal space. This proves that the left-behind fingerprint belongs entirely to the optimization method itself, remaining completely invariant to the specific context or data being deleted. 

**BrainCode: Turning Human-Agent Communication Into a Programming Language (men-** 

**tored by researcher Guy Dar)** In AI agents, human requests and agent thinking are described in unstructured natural language. On the other hand, programming languages are concise and structured, but they lack the expressive power of natural language. In this project, we aim to bridge this gap by turning natural language into a programming language, just very rich and “long-tailed”. To achieve this, we operate in a semi-automated iterative process, where AI develops meta-concepts in the language, through AI debate, based on a large batch of real-world examples, and corrected by humans until reaching an acceptable outcome. The language should be rich enough to express human requests and agentic thinking but without redundancy and no dependence on the linguistic style of the specific model. 

This allows for standardisation that in future work can help identify points of difficulty for AI, and in some cases improve them. 

**Improving Sent2model to production level (mentored by researcher Guy Dar)** In a recent paper [2], we proposed a new agentic system. Given a natural language question about data without a dataset, the agent automates the process of data collection and looks for data across online resources. It creates a new structured dataset for the question, and, if requested, the agent also creates a statistical model trained on the data. The agent is implemented as an AI skill and works reasonably well. In this 

Page 8 of 9 

Natural Language Processing 

Final Project: Instructions 

project, we aim to improve the robustness and complexity of questions sent2model can answer. This process requires a process of human/automated prompt optimization, engineering efforts to handle the smooth run of scraping and queries, extend the workflow where possible in creative ways (e.g. new sources of data, deeper analysis), and evaluating the success of the process. 

# **References** 

- [1] Y. Chen, S. Pal, Y. Zhang, Q. Qu, and S. Liu. Unlearning isn’t invisible: Detecting unlearning traces in llms from model outputs. _arXiv preprint arXiv:2506.14003_ , 2025. 

- [2] Y. Einy, G. Dar, S. Novgorodov, T. Milo, and N. Frost. Automating efficient data collection through the synergy of agentic ai and active learning, 2026. 

- [3] D. Gottesman, A. Gilae-Dotan, I. Cohen, Y. Gur-Arieh, M. Mosbach, O. Yoran, and M. Geva. Lment: A suite for analyzing knowledge in language models from pretraining data to representations. _arXiv preprint arXiv:2509.03405_ , 2025. 

- [4] C. Haya, O. Shafran, and M. Geva. Don’t forget your embeddings: Robust knowledge erasure via precise editing of embeddings. _arXiv preprint arXiv:2606.03695_ , 2026. 

- [5] N. Kosmyna, E. Hauptmann, Y. T. Yuan, J. Situ, X.-H. Liao, A. V. Beresnitzky, I. Braunstein, and P. Maes. Your brain on chatgpt: Accumulation of cognitive debt when using an ai assistant for essay writing task. _arXiv preprint arXiv:2506.08872_ , 4, 2025. 

- [6] R. Menary. Writing as thinking. _Language sciences_ , 29(5):621–632, 2007. 

- [7] K. O’Brien, S. Casper, Q. Anthony, T. Korbak, R. Kirk, X. Davies, I. Mishra, G. Irving, Y. Gal, and S. Biderman. Deep ignorance: Filtering pretraining data builds tamper-resistant safeguards into open-weight llms. _arXiv preprint arXiv:2508.06601_ , 2025. 

- [8] O. Shafran, S. Ronen, O. Fahn, S. Ravfogel, A. Geiger, and M. Geva. From directions to regions: Decomposing activations in language models via local geometry. In _Forty-third International Conference on Machine Learning_ , 2026. 

- [9] S. Singh, S. Ravfogel, J. Herzig, R. Aharoni, R. Cotterell, and P. Kumaraguru. Representation surgery: theory and practice of affine steering. In _Proceedings of the 41st International Conference on Machine Learning_ , ICML’24. JMLR.org, 2024. 

- [10] J. T.-Z. Wei, A. Godbole, M. A. Khan, R. Wang, X. Zhu, J. Flemings, N. Kashyap, K. P. Gummadi, W. Neiswanger, and R. Jia. Hubble: a model suite to advance the study of llm memorization. _arXiv preprint arXiv:2510.19811_ , 2025. 

- [11] Z. Wu, A. Arora, A. Geiger, Z. Wang, J. Huang, D. Jurafsky, C. D. Manning, and C. Potts. Axbench: Steering LLMs? even simple baselines outperform sparse autoencoders. In _Forty-second International Conference on Machine Learning_ , 2025. 

Page 9 of 9 


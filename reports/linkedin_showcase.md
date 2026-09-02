# LinkedIn Showcase Post: TransChemo-Repurpose 🚀🧬

*Copy and paste the text below directly to your LinkedIn post. Attach screenshots or a video recording of the interactive Streamlit dashboard!*

---

🚀 Excited to share my latest bioinformatics & AI in drug discovery project: **TransChemo-Repurpose** — A Multi-Modal Deep Learning & Explainable AI framework for precision oncology drug repurposing! 🧬💊

Developing a new de novo drug takes over **12 years, costs >$2.5 Billion**, and faces a **>90% failure rate** in clinical trials. Drug repurposing offers a powerful alternative, but traditional transcriptomics-based matching methods (like classical CMap Kolmogorov-Smirnov heuristics) cannot model complex non-linear chemistry-gene interactions and fail to generalize to novel chemical scaffolds.

To bridge this gap, I designed and implemented **TransChemo-Repurpose**, an end-to-end multi-modal AI platform unifying patient transcriptomics with molecular chemoinformatics.

---

### 🔬 What makes this architecture unique?

1️⃣ **Multi-Modal Cross-Attention Network (PyTorch):**
Fuses high-dimensional cancer gene expression vectors (LINCS L1000 landmark space) with 2048-bit Morgan Fingerprints (ECFP4) and ADMET physicochemical descriptors to predict **Signature Reversal Scores (SRS)**, **Bioactivity ($pIC_{50}$)**, and **Phenotypic Efficacy**.

2️⃣ **Zero Chemical Data Leakage (Bemis-Murcko Scaffold Splitting):**
To ensure genuine out-of-distribution evaluation, models were evaluated on scaffold-split 5-fold cross-validation.
📊 **Results:**
• **AUROC:** 0.894 (vs. 0.825 for XGBoost, 0.684 for Classical CMap)
• **Pearson $r$:** 0.864 (a **+65.8% increase** in reversal correlation over standard CMap)
• **Enrichment Factor (EF@10%):** 4.12x active hit recovery

3️⃣ **Explainable AI (XAI) & Pathway Biology:**
Integrated **SHAP** to identify pharmacophoric chemical moieties driving efficacy and performed **Gene Set Enrichment Analysis (GSEA)** to validate systemic restoration of hallmarks like E2F cell cycle targets and EMT in Triple-Negative Breast Cancer (TNBC) and Glioblastoma (GBM).

4️⃣ **Interactive Portfolio Dashboard (Streamlit + Plotly + RDKit):**
Enables real-time virtual screening, dynamic volcano plots, 2D SVG molecular structure rendering, and pathway perturbation tracking.

---

💻 **GitHub Repository & Full Technical Paper:** [Insert your GitHub repo link here]
📄 **Complete Research Whitepaper:** Check the `reports/technical_report.md` in the repo!

I'm actively preparing for my **Master's in Computational Biology / Bioinformatics / AI in Drug Discovery**, and I'd love to connect with researchers, professors, and industry professionals working at the intersection of AI and life sciences!

What are your thoughts on multi-modal representation learning in computational pharmacology? Let's discuss in the comments! 👇

---

#Bioinformatics #ComputationalBiology #MachineLearning #DeepLearning #DrugDiscovery #Chemoinformatics #Transcriptomics #RDKit #PyTorch #Streamlit #CancerResearch #PrecisionMedicine #XAI #ArtificialIntelligence

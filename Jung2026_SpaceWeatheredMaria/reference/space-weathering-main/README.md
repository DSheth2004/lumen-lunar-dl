# space-weathering
This repository provides a **space weathering correction** based on **Hapke-based spectral unmixing framework**. The modified MCMC algorithm is based on **Lapotre et al. (2017)**.

## **Workflow**
The repo consists of three main steps:

### **1️⃣ Step 1: Space Weathering Depth Correction**
- **Directory:** `hapkeRTM/hapkeRTM.py`
- **Class:** `HapkeCorrection`
- **Example Notebook:** [`hapkeRTM/examples.ipynb`](hapkeRTM/examples.ipynb)

### **2️⃣ Step 2: MCMC-Based Spectral Unmixing**
- **Directory:** `mcmc/`
- **Required Scripts:** `load_data.m`, `MAIN.m`
- **Original Reference:**
  - Lapotre, M.G.A., B.L. Ehlmann, and S.E. Minson (2017),  
    *A probabilistic approach to remote compositional analysis of planetary surfaces*,  
    JGR: Planets, doi: [10.1002/2016JE005248](https://doi.org/10.1002/2016JE005248).

This step applies a **Markov Chain Monte Carlo (MCMC) approach** to estimate mineral abundances from spectral data. The original MCMC model's prior setting has been **modified** for enhanced spectral unmixing tailored for lunar mare basalt applications.

---

### **3️⃣ Appendices C & D**
- **Directory:** `hapkeRTM/hapkeRTM.py`
- **Class:** `HapkeRTM`

📌 **Appendix C:** Derivation of single scattering albedo (SSA)  
📌 **Appendix D:** Calculation of mineral endmember's imaginary refractive index  

An example for **olivine** is available in:  
[`hapkeRTM/examples.ipynb`](hapkeRTM/examples.ipynb)

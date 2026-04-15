# Physics-Embedded End-to-End Differentiable Framework (PE-E2ED)

Official implementation of the paper:  
**"Physics-Embedded End-to-End Differentiable Framework for High-Precision and Robust Interferometry"**

[![Paper](https://img.shields.io/badge/Paper-In__Press-red)](https://github.com/srzzz76/PE-E2ED)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Code](https://img.shields.io/badge/Code-Coming__Soon-yellow)](https://github.com/srzzz76/PE-E2ED)


## 📢 News
* **[2026.04]** The paper has been submitted. 
* **[Important]** The source code is currently private and will be fully released upon official publication. Feel free to **Star** this repository for updates.

---

## 💡 Overview

Interferometric phase retrieval is fundamental to high-precision optical metrology. However, reconciling the computational efficiency of deep learning with the rigorous interpretability of physical models remains a challenge.

**PE-E2ED** introduces a physics-embedded, end-to-end differentiable framework that internalizes a classical physical solver as a functional layer. This architecture enables global optimization while preserving gradient propagation, effectively overcoming the non-differentiable bottleneck of traditional phase unwrapping. Consequently, our framework achieves state-of-the-art (SOTA) performance in high-precision nanoscale metrology:

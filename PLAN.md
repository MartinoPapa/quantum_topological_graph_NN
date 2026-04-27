# Plan for Abstract and Introduction

I will structure the abstract and introduction sections based on the requirements and the details provided in `readme.md`.

1. **Creating the Abstract (`tex/sections/abstract.tex`)**:
   - I will create a new file for the abstract using `\chapter*{Abstract}`.
   - **Content**: 
     - Briefly introduce the overarching challenge of fraud detection.
     - State the primary goal of the thesis: to develop a detailed Python implementation of a Quantum Topological Graph Neural Network (QTGNN).
     - Very briefly mention the datasets used for evaluation to test the model across various structural challenges: the Elliptic Bitcoin Dataset, Ethereum Phishing Transaction Networks, IBM AMLSim, and PaySim. 
   - I will update `tex/thesis.tex` to include `\include{sections/abstract}` just before the `\tableofcontents`.

2. **Writing the Introduction (`tex/sections/introduction.tex`)**:
   - **Context & Importance**: I will start by explaining the critical importance of fraud detection (preventing massive economic losses, disrupting illicit organizations).
   - **The Growing Challenge**: I will address why it's becoming more difficult to detect fraud. I'll highlight the sheer volume of digital transactions and the rise of cryptocurrencies/Web3 which offer pseudo-anonymity. I will detail increasingly sophisticated tactics used by criminals, such as complex fraud rings, cyclic transfers, and **mixers** (providing a brief explanation that mixers are services that pool together and randomly redistribute cryptocurrency to break the traceable link between sender and receiver).
   - **Proposed Solution (QTGNN)**: I will briefly introduce how traditional heuristic methods fall short and why the QTGNN (combining Quantum Computing, Graph Theory, and Topological Data Analysis) offers a robust solution capable of capturing these hidden patterns.
   - **Thesis Structure/Objectives**: I will conclude the introduction by outlining the thesis objectives: developing the Python implementation, training it on a quantum simulator, testing on an actual quantum computer to reduce costs, and evaluating its performance across the aforementioned real-world and synthetic datasets.

Please approve this revised plan so I can proceed with writing the LaTeX code.

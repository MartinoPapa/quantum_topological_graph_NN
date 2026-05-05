# Plan

1. **Fix spelling and grammatical errors in `tex/sections/abstract.tex`**:
   The sentence provided contains a few typographical and phrasing issues. I will correct them as follows:
   - "Laveraging" -> "Leveraging"
   - "decisions taken by the model" -> "decisions made by the model"
   - "extreamly" -> "extremely"
   - "in regard to use deep learning model" -> "in regard to using deep learning models"
   - "compleatly" -> "completely"
   - "In a legal process the model needs to be able to explain the decisions taken by it." -> "In a legal setting, the model must be able to explain its decisions."

   The revised sentence will be:
   > "Leveraging these features we will also be able to explain the decisions made by the model. This is extremely important since, in regard to using deep learning models in the real world, we must avoid treating them as completely black boxes. In a legal setting, the model must be able to explain its decisions \cite{goodman2017european}."

2. **Add a relevant citation to `tex/refs.bib`**:
   I will add a BibTeX entry for a foundational paper that discusses the legal necessity of explainability in AI models, specifically regarding the "right to explanation":
   - **Goodman, B., & Flaxman, S. (2017).** *European Union regulations on algorithmic decision-making and a "right to explanation".* AI Magazine.
   - This paper strongly supports the claim that in real-world and legal scenarios, models cannot operate as complete black boxes.

3. **Apply the citation**:
   I will append `\cite{goodman2017european}` to the end of the corrected sentence in the abstract.

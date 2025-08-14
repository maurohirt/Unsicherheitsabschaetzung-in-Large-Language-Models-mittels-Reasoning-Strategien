# Unsicherheitsabschätzung in Large Language Models mittels Reasoning-Strategien

Dieses Repository bündelt Code und Auswertungen zur Frage, wie Reasoning-Strategien in LLMs zur Quantifizierung epistemischer Unsicherheit auf Antwortebene genutzt werden können.

## Kurzüberblick
- __Reasoning-Strategien__: Chain-of-Thought (CoT), Chain-of-Draft (CoD), Tree-of-Thought (ToT).
- __UQ-Verfahren__: Self-Evaluation (P-True, Self-Probing) sowie tokenwahrscheinlichkeitsbasierte Aggregationen (Probas-Mean, Probas-Min, Token-SAR) – jeweils als Final-Answer-Baselines und in reasoning-bewussten Varianten mit Keyword-Selektion und Importance-Gewichten entlang der Reasoning-Kette.
- __Benchmarks & Modelle__: HotpotQA, 2WikiMHQA, GSM8K, SVAMP, ASDiv (Llama-3.1-8B). ToT-Setting: Game of 24 (BFS) mit DeepSeek-V3-Chat; UQ-Signale dienen ausschließlich der internen Pfadauswahl.
- __Bewertung__: AUROC, ECE, Reliability Curves, Accuracy.
- __Zentrale Ergebnisse__:
  - Keyword-gewichtete Aggregationen verbessern Diskrimination und Kalibration konsistent; besonders stark: Probas-Min und Token-SAR. Keyword-Selektion reduziert Längen-/Formatbias; Importance-Gewichte erhöhen Sensitivität für entscheidungsrelevante Tokens.
  - CoD erhält die UQ-Qualität weitgehend bei geringerem Tokenbudget, mit ca. −2.84 Prozentpunkten Accuracy gegenüber CoT.
  - Self-Evaluation bringt begrenzten Mehrwert: P-True zeigt engen Konfidenzbereich, Self-Probing schwache Kopplung zwischen Confidence und Korrektheit.
  - ToT-Kostenbefund: Wechsel von GPT-4 auf DeepSeek senkt Baselinekosten ~99%; UQ-Branch-Scores reduzieren weitere 3–5×, gehen jedoch mit deutlichen Genauigkeitseinbußen einher (z. B. 72% → bis 24%, Probas-Mean, T≈1.8).

## Repository-Übersicht

Dieses Repository enthält drei Hauptordner. Unten findest du eine kurze Beschreibung sowie Verweise auf die detaillierten READMEs in den Ordnern.

- __CoT-UQ__
  - Kurz: CoT und CoD Projektcode, Skripte und Experimente.
  - Details: [CoT-UQ/README.md](CoT-UQ/README.md)

- __Results_Analysis__
  - Kurz: Auswertung und Visualisierung von Ergebnissen (z. B. Notebooks, Plots, Tabellen).
  - Details: [Results_Analysis/README.md](Results_Analysis/README.md)

- __tree-of-thought-llm__
  - Kurz: Implementierung/Experimente zum Tree-of-Thought-Ansatz (u. a. Game24-Beispiel).
  - Details: [tree-of-thought-llm/README_GAME24.md](tree-of-thought-llm/README.md)

Hinweis: Für Setup, Nutzung und weiterführende Informationen bitte jeweils das README im entsprechenden Ordner lesen.

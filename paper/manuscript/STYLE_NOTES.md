# Style notes from recent accepted evaluation / analysis papers

Looked at (modest effort, three papers of our kind, not method papers):

1. Edwards and Camacho-Collados (2024). Language Models for Text Classification: Is In-Context
   Learning Enough? LREC-COLING 2024. https://aclanthology.org/2024.lrec-main.879/
2. Min et al. (2022). Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?
   EMNLP 2022. https://aclanthology.org/2022.emnlp-main.759/
3. Pan et al. (2023). What In-Context Learning "Learns" In-Context: Disentangling Task Recognition
   and Task Learning. Findings of ACL 2023. https://aclanthology.org/2023.findings-acl.527/

## What they do

- Abstract: 150-200 words (Edwards ~190, Min ~170, Pan ~200). One sentence of context, one of gap,
  one of what was done (scale: "16 datasets", "12 models"), then the findings as a short numbered
  or "In general" sentence. No numbers with decimals in the abstract.
- Introduction: ends with the findings or contributions as an explicit list. Min: "We find that:
  (1) ... (2) ... (3) ..." followed by "In summary, our analysis ... (Section 4) ... (Section 5)".
  Pan: two bullets, each a finding with its qualifier. Edwards: "Our main contributions are as
  follows. First, ... Second, ... Third, ...". Each item is a finding with its scope, not a
  description of an activity. Section pointers are attached to the items.
- Related work: short thematic paragraphs with a one-line opener each (Min), or numbered
  subsections (Edwards). Pan puts Related Work after the results. Each paragraph ends with one
  sentence that positions the paper against the group ("Our question is a different one ...").
- Setup: numbered subsections Datasets / Models / Task Setup (Pan) or run-in bold headings
  "Models." (Min). Short prose, a table for datasets or models, hyperparameters and dataset
  references pushed to the appendix. Subsample sizes and the reason for them are stated in one
  sentence ("We use fewer examples due to budget constraints").
- Results: every subsection opens with the claim as a plain sentence, then the figure or table
  pointer, then the numbers. Pan 4.2: "The trends for task learning generalize across different
  types of abstract labels. In Figure 3, we show ...". Min uses a run-in "Results." paragraph
  after describing the variant being tested.
- Limitations: one or two plain paragraphs, 100-200 words, listing scope limits (task type,
  languages, model sizes, number of prompts) and what remains unexplained, each with one clause of
  future work. No defensive tone.

## What we adopted

- Abstract cut to 150 words, no decimals, findings in four short sentences.
- Introduction ends with an enumerated list of four contributions, each stating a finding and its
  scope, with a pointer to the protocol section.
- Related Work as four paragraphs with bold run-in headings (\paragraph), each closing with one
  sentence that positions our study against that group.
- Setup as five numbered subsections (Datasets, Models, Protocol, Methods Compared, Metrics and Statistics), one table
  for the datasets, one enumerated list for the five protocol fixes, hyperparameters in one
  sentence each, prompt template and normalisation rules in Appendix A.
- Results skeleton: each subsection heading states the claim ("Narrowing the label set helps a
  little, and reliably"), and the prose draft already opens each subsection with the claim before
  the numbers.
- Limitations planned as plain prose from the bullet list in sections/limitations.tex.

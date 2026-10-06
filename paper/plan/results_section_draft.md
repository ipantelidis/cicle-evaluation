# Draft: Experimental Setup (protocol, baselines, statistics) and Results

Numbers: six small LLMs x three seeds at the reference setting (MiniLM, logistic regression,
alpha = 0.05) unless stated. Sources are given in brackets after each paragraph:
A6 = `paper/plan/analysis_6models_permodel_bootstrap200.txt`, A8 =
`paper/reference/analysis_2026-10-06_morning.txt`, F = findings file, R = per-instance result
files (new analysis, see `figure_scripts_todo.md`). Bootstrap CIs are from 200 resamples; redo
with 5,000 before submission. [PENDING: ...] marks results still running. Word counts are at the
end of each subsection; the running total covers everything from 4.3 onwards.

---

## 4.3 Protocol

Every experiment uses a stratified subsample of 2,000 training and 1,000 test instances per
dataset, drawn with three seeds; 400 of the 2,000 are held out for conformal calibration, so the
base classifier, the retrieval pool and the fine-tuned baselines all see the same 1,600 labelled
examples. The subsample keeps the cost of 18 LLM runs per configuration tractable while leaving the
1,000-instance test sample large enough for the paired tests in Sec. 4.5; the seeds capture the
sampling noise that a single split hides.

Our earlier evaluation of this pipeline [DS2026 submission] differed from the present one in three
respects that turned out to drive its conclusions. First, only the zero-shot prompt listed the
allowed labels; few-shot and CICLe prompts did not, and used different wording. Between 20% and
35% of the outputs of some models then matched no label and were scored as errors, which made
CICLe appear to hurt Qwen-2.5 and Ministral. Second, the few-shot baseline retrieved from a
different example pool than CICLe. Third, the conformal sets were not seeded, so CICLe runs could
not be paired with their few-shot counterparts instance by instance. The corrected protocol (i)
lists the admissible labels in every prompt (all classes for zero- and few-shot, the candidate set
for narrowing methods) with identical wording otherwise, so that the candidate list and the
retrieved examples are the only difference between conditions; (ii) retrieves for every method from
the same 1,600-example pool; (iii) seeds the calibration split, the conformal sets and the test
sample; (iv) sets the generation budget from the longest label and normalises the output (case,
quotes, punctuation, trailing text) before matching it to a label, counting anything that still
matches no label as an invalid output and reporting its rate; and (v) saves a per-instance record
(gold label, raw output, prediction, candidate set, retrieved example ids, prompt length) for every
run. Invalid outputs fell to 0-6% averaged over models (zero-shot: 6.1% Yahoo, 5.1% SemEval, 5.2%
Ohsumed, 3.0% GoEmotions, 0.0% SST; CICLe 1.2 / 4.5 / 1.5 / 2.2 / 0.0%), with Qwen-2.5-3B still
the worst case at up to 24.5% on SemEval-18 (Appendix G2). [A6 tables; R metrics]

Two retrieval variants are run for every prompting method. *Fixed* places the k training examples
most similar to the input (cosine similarity in the embedding space) that belong to an admissible
class; *Per-Class* (PC) places the k most similar examples of each admissible class, so the number
of examples is k x |candidates| and shrinks with the candidate set. k ranges over {1, 2, 4, 8} on
the four short-text datasets and {1, 4} elsewhere. Ohsumed abstracts average about 190 words, so
only Fixed is run there [PENDING: PC k = 1]. (443 words)

## 4.4 Methods compared

*Zero-shot* lists all labels and no examples. *Few-shot* adds retrieved examples from all classes.
*CICLe* restricts the label list and the retrieval to the class-conditional conformal set at
miscoverage alpha; a singleton set is returned without calling the LLM, an empty set falls back to
the base classifier's top label. To separate the contribution of conformal calibration from that
of narrowing as such, three further narrowing rules use the same base classifier and the same
prompt: *top-k* keeps the m most probable classes, where m is the mean conformal set size on that
dataset, so the average budget is identical but the set does not adapt per instance;
*probability mass* keeps the most probable classes until their (uncalibrated) probabilities sum to
1 - alpha, which adapts per instance but trusts the classifier's confidence; *marginal CP*
calibrates one threshold over all classes instead of one per class [PENDING]; an *oracle* set
contains the gold label plus random classes at the conformal budget and gives the ceiling for any
narrowing rule at that set size [PENDING]. Two supervised baselines use no LLM: the base classifier
itself (MiniLM embeddings + logistic regression; TF-IDF, Contriever and an SVM in the ablations) and
RoBERTa-base fine-tuned on the same 1,600 examples, with the epoch chosen on the 400 calibration
examples. Both see exactly the labelled data the LLM pipelines see. (224 words; running 667)

## 4.5 Metrics and statistics

We report macro-F1 over the classes present in the test sample, because several datasets are
long-tailed and because macro-F1 is what the narrowing rules differ on (Sec. 5.3); accuracy is in
Appendix B. Every comparison between two methods is paired: the same model, seed, k and variant,
hence the same 1,000 test instances, the same retrieved pool and the same conformal sets. We report
the mean paired difference in macro-F1 over all such cells and a 95% confidence interval from a
bootstrap that resamples test instances and applies the same resample to both methods and to all
cells sharing that seed. The interval therefore reflects test-sample noise, not the spread over
hyperparameters that the earlier version's error bars showed; n is the number of (model, seed, k)
cells, 72 for the main grid (6 models x 3 seeds x 4 values of k) and 36 where k is in {1, 4}. We
also report mean prompt tokens per LLM call and the share of instances the LLM never sees.
(170 words; running 837)

---

## 5 Results

### 5.1 The protocol decides the conclusion

Under the corrected protocol the model-level picture of the earlier version reverses. There,
CICLe with Fixed retrieval appeared to cost Ministral-3B 2.1 points and Qwen-2.5-7B 3.0 points on
Yahoo Answers; here the same comparison gives -0.17 [95% CI -1.24, +1.02] and +0.05 [-0.47,
+0.59], neither distinguishable from zero over 12 paired cells, and with Per-Class retrieval
every one of the six models gains or is unchanged (Appendix D). The difference is almost entirely
the label list: a prompt that names the admissible labels removes most unparseable outputs, and
those had been concentrated in exactly the models that appeared to be hurt (Appendix G2). The
pairing matters as well: with the seeded conformal sets, the paired bootstrap narrows the
interval around a +1.4-point Per-Class gain on Yahoo to +-0.4 points, where unseeded runs over the
same grid would not have been comparable instance by instance. We take two lessons into the rest of
the section: report invalid outputs, and never compare two prompting conditions that differ in
more than the quantity of interest. [A6 per-model block; DS2026 Table 1] (172 words; running 1,009)

### 5.2 Narrowing the label set helps a little, and reliably

Table 2 gives macro-F1 at k = 4 for every method and the paired difference between CICLe and
few-shot pooled over all k; Figure 1 plots every configuration against its prompt length. CICLe
improves on few-shot prompting in eight of nine dataset x variant cells, by between +0.5 and +1.8
points: Yahoo PC +1.40 [+1.03, +1.76]; SST-5 Fixed +1.77 [+1.04, +2.49] and PC +1.07 [+0.36,
+1.69]; SemEval-18 +0.52 [+0.31, +0.71] and +0.46 [+0.27, +0.68]; GoEmotions +0.59 [+0.24, +1.07]
and +0.92 [+0.47, +1.37]; Ohsumed Fixed +1.15 [+0.60, +1.86] (72 pairs each, Ohsumed 36). The one
exception is Yahoo with Fixed retrieval, -0.19 [-0.59, +0.17], where the balanced ten-class
problem leaves the conformal set with 5.7 of 10 classes on average and a single retrieved example
already carries the label. These are small effects: all intervals lie within +-2.5 points, and the
gain does not grow with k. At k = 8 CICLe is still ahead on SST (46.8 vs 45.8 Fixed; 47.1 vs 46.3
PC), SemEval (16.1 vs 15.4; 16.0 vs 15.7) and GoEmotions Fixed (27.3 vs 26.6), and behind on Yahoo
Fixed (59.7 vs 59.8), so the picture at large k is the pooled one, not a convergence. [A6 tables
and paired tests]

The second effect is on the prompt. Because Per-Class retrieval places k examples per admissible
class, narrowing the list shortens the prompt in proportion: at equal k CICLe PC uses 28-38% fewer
tokens than few-shot PC on Yahoo (1,378 vs 2,166 at k = 4), 16-21% on GoEmotions, 11-20% on SST
and 11-14% on SemEval, while scoring the same or higher (Figure 1, filled markers). The Fixed
variant is cheaper still, and on GoEmotions it is also the better one: averaged over k, CICLe Fixed
reaches 26.6 against 25.9 for CICLe PC at about a sixth of the tokens (288 vs 1,788), because 22
admissible classes x k examples dilute the prompt more than they inform it. On Yahoo and SST the
order is the reverse (PC +4.5 and +1.8 points over Fixed at k = 4). Zero-shot prompting is within half a
point of one Fixed example on Yahoo, SST and GoEmotions, and 2 and 11 points behind it on SemEval
and Ohsumed; the few-shot gains come with k and with the Per-Class budget, not from the first example.
[PENDING: k = 0, the candidate list with no examples, isolates how much of the gain is the list
itself.] (378 words; running 1,387)

### 5.3 How you narrow matters

Narrowing could help simply because a shorter label list is easier to choose from. If so, any
rule that produces sets of the same size should do. Table C1 and Figure 2 test this with top-k
and probability-mass sets built from the same classifier at the same mean size. On the two
balanced datasets the three rules are interchangeable: on SST-5 CICLe minus top-k is -0.14 [-0.84,
+0.67] Fixed and +0.32 [-0.50, +1.08] PC, minus mass +1.07 [+0.46, +1.71] and +0.71 [+0.14,
+1.26]; on SemEval-18 every difference is below 0.05 points (36 pairs each). On the naturally
long-tailed sets conformal calibration is worth two points or more: GoEmotions +2.51 [+1.51,
+3.39] over top-k and +2.26 [+1.40, +2.97] over mass (Fixed; PC +2.54 and +1.91), Ohsumed +1.08
[+0.41, +1.76] and +1.52 [+0.84, +2.13]. [A6]

The controlled experiment pins the cause on the labelled pool. We keep the 1,000 test instances of
Yahoo and SST fixed and replace the stratified training sample with a long-tailed one whose
largest class is 10x or 100x the smallest (five examples). The conformal sets keep their
coverage: 95.0 -> 94.9 -> 96.7% on Yahoo and 95.4 -> 96.5 -> 96.0% on SST at mean sizes of
5.7-7.7 and 3.9-4.4 classes. Top-k sets of the same size fall to 93.7 -> 90.5 -> 81.6% and 95.5 ->
89.4 -> 88.4%; probability-mass sets to 98.4 -> 96.2 -> 74.2% and 99.2 -> 92.2 -> 60.0%, shrinking
as the classifier grows over-confident on the head classes (Figure 2b). The missing coverage is
all on the tail: in the seed-42 Yahoo run at 100x, the top-k set contains the gold label for 22%
of the rarest class's test instances and the mass set for 2%, against 93-100% for every class under
CP (Figure 2c). An LLM cannot recover a label it was not offered, so the macro-F1 gap opens
accordingly: at 100x, CICLe minus top-k is +6.52 [+5.82, +7.27] Fixed and +7.44 [+6.76, +8.21] PC
on Yahoo, +3.35 [+2.50, +4.06] and +4.21 [+3.39, +5.01] on SST; minus mass +9.08 [+8.34, +9.76]
and +9.44 [+8.71, +10.12] on Yahoo, +8.54 [+7.51, +9.54] and +10.70 [+9.60, +11.68] on SST (36
pairs each). At 10x the gaps are +0.3 to +2.4 points (the smallest, Yahoo Fixed vs mass, not significant). [PENDING: marginal CP, which calibrates one
threshold for all classes, covers 62.9% of gold labels at a mean size of 4.1 on Yahoo 100x and is
expected to land between top-k and mass once the LLM runs finish.] [PENDING: the oracle set at the
same budget gives the ceiling.] [A8 paired tests and candidate-set lines; R per-class coverage]

Two things do not happen. CICLe's margin over few-shot does not grow with imbalance: on Yahoo PC
it is +1.32 at 1x (k in {1, 4}), +1.09 [+0.68, +1.53] at 10x and +0.89 [+0.55, +1.24] at 100x; on
SST +1.14, +1.82 [+1.18, +2.52] and +1.16 [+0.54, +1.78]. Few-shot prompting lists every label
and so cannot lose coverage; what imbalance degrades is its retrieved examples, and the conformal
set has the same problem. The earlier version's claim that imbalance predicts CICLe's benefit
rested on four datasets that differed in more than their imbalance; the controlled comparison does
not support it. Second, narrowing is not free of harm. Pairing instances between CICLe PC and
few-shot PC at k = 4 (18,000 pairs per dataset), CICLe fixes 4.7% of Yahoo instances and breaks
3.4%; among the 5.0% of instances whose gold label the conformal set misses, it fixes none and
breaks 4.1% that few-shot had right (GoEmotions: 4.6% fixed, 3.2% broken, 8.3% broken among the
4.6% uncovered). The 1 - alpha coverage is thus a direct bound on the damage, which is the
argument for calibrating per class in the first place. [A8; R, Appendix G1] (564 words; running 1,951)

### 5.4 Uninformative label names

Few-shot prompting with informative label names can succeed without reading the examples: the
word "Sports" is a strong hint. To remove that hint while keeping the task, we rename every class
to a nonsense word, identically for all seeds, on Yahoo and GoEmotions; the base classifier is
untouched. Few-shot accuracy collapses when the examples are few: with one Fixed example on Yahoo
it scores 25.3 macro-F1 against 54.8 with the original names, while CICLe keeps 34.3 (Table 3).
The paired advantage of CICLe is the largest we measure anywhere, +6.52 [+5.89, +7.08] Fixed and
+5.62 [+5.02, +6.25] PC on Yahoo (36 pairs), and remains positive on GoEmotions (+0.66 [+0.31,
+0.98]; +0.98 [+0.65, +1.39]), where the original names were already weak hints (+0.59 / +0.92 with the
original names). A shorter, classifier-ranked list of candidate labels is most useful
exactly when the label names themselves tell the model nothing. (148 words; running 2,099)

### 5.5 When is the LLM pipeline worth it?

Table 4 puts the pipelines next to two models that never call an LLM. With 1,600 balanced labels
a fine-tuned RoBERTa-base beats the best six-model-average pipeline on four of five datasets:
51.2 vs 47.6 on SST-5, 24.5 vs 16.1 on SemEval-18, 37.0 vs 27.3 on GoEmotions and 55.4 vs 51.7 on
Ohsumed. On Yahoo the picture is starker: the logistic regression that CICLe narrows with scores
65.4 on its own, above every LLM pipeline (best 63.8, CICLe PC k = 8) and above RoBERTa (61.8).
In that regime the pipelines buy nothing. The order flips when the labelled pool is skewed. At 100x
imbalance RoBERTa falls to 41.2 on Yahoo and 31.9 on SST and the logistic regression to 31.4 and
12.6, while the best pipeline loses only 7.8 and 4.2 points (56.0 and 43.4; CICLe PC k = 1) because
retrieval and the LLM's prior knowledge of the labels carry the rare classes; the few-shot pipeline
without narrowing is close behind (55.1 and 42.1). At 10x the pipelines and RoBERTa are within
two points on both datasets. [PENDING: pool sizes 250 / 500 / 1,000, which test whether a small
balanced pool behaves like a skewed one.] The practical rule is therefore not "use CICLe" but
"use a classifier when the labelled data are plentiful and balanced; use an in-context pipeline
when they are skewed [or scarce], and narrow with class-conditional CP when you do". (216 words; running 2,315)

### 5.6 Choosing alpha; models and model sizes

Alpha sets the coverage of the candidate set and thereby its size and the share of instances
the classifier answers alone (Figure 3). Loosening it from 0.05 to 0.20 on Yahoo turns a -0.4
difference to few-shot into +3.6 while 35% of test instances never reach the LLM; on GoEmotions
and SemEval, where sets at 0.05 hold 22 of 28 and 17 of 20 classes, it adds +3.0 and +1.4; on
SST the gain is flat at +2.5 to +2.8 for every alpha >= 0.05 (two Llama models, 24 cells). The
loosening pays where the base classifier is strong enough that its singleton answers are right
(Yahoo) or where the sets are otherwise too large to help (GoEmotions); the conformal coverage
guarantee, not accuracy, is what one gives up. Among the other components, the embedding matters
more than the classifier: Contriever beats MiniLM on SST (+3.7 vs +2.7 over few-shot) and
GoEmotions (+2.5 vs +0.9) but gives lower absolute macro-F1 on Yahoo (57.8 vs 59.5); TF-IDF is
weakest everywhere; SVM and
logistic regression are within 0.4 points (Appendix E). Across models, Per-Class CICLe never
significantly hurts any of the six; Fixed CICLe hurts Llama-3.2-3B (-2.33 [-3.12, -1.60]) and
Mistral-7B (-0.69 [-1.21, -0.17]) on Yahoo only (Appendix D). The effect carries to larger
models: CICLe adds +0.1 to +4.4 points to Mistral-Nemo-12B and Qwen-2.5-32B under the same
protocol (Appendix F; 32B on one seed [PENDING seeds 43/44]), and a 3B model with CICLe PC at k = 4
matches Qwen-32B zero-shot on the same seed (61.5 vs 60.7 on Yahoo; 51.0 vs 48.8 on SST), which
is the form in which the earlier version's size comparison survives. [A6/A8 ablation blocks; A6
per-model; R large-model files] (450 words; running 2,765)

---

Running total from 4.3: 2,765 words (counted without the source brackets), of which Results
(5.1-5.6) = 1,928 and Setup (4.3-4.5) = 837. Results is ~270 words under the 2,200 target, which
is the room reserved for the pending pool-size paragraph in 5.5 (~120 words), the marginal-CP and
oracle sentences in 5.3 (~60 words) and the k = 0 sentence in 5.2 (~40 words). Setup 4.3 is long
(443 words) because it carries the protocol-fix argument; if the page budget bites, move the
enumerated list (i)-(v) into a compact table in Appendix A and keep two sentences. 5.6 (450 words)
is the other cut candidate: the per-model and large-model sentences can move entirely to the
appendices.

Open numeric items before the text is final
1. Replace all bootstrap-200 intervals with bootstrap-5000 (run `analyze.py --bootstrap 5000
   --per-model`, six models, all 11 variants; several minutes).
2. The 1x points in Figure 2a and the 5.3 sentence "+1.32 at 1x" use k in {1, 4} to match the
   10x/100x runs; the CI for that restricted set must be computed (A6 only tests all four k).
3. Ohsumed PC k = 1 cell [PENDING].
4. Mean input length for Ohsumed in tokens (Table 1).
5. Decide whether SST and SemEval renaming are worth running (not planned; the text does not
   need them).

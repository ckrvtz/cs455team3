# Scoring rubric for prompt evaluation

How to score the completions that `bin/eval` writes to
`runs/<run>/samples_<step>.txt` (ROADMAP P6.2, P6.3; proposal: manual 0–5
scoring on grammar, consistency and sense). Scores go in `eval/scores.csv`.

## What to score

- **Which completion:** score the **temperature 0.8** completion of each
  prompt (the second section of each block). The greedy completion is kept
  for comparing builds, not for scoring. Greedy decoding of small models
  tends to loop and would mostly measure that.
- **What counts:** only the continuation, read as a continuation of the
  prompt. The prompt text itself is not scored.
- **Truncation:** completions stop after 120 tokens. Do not penalize an
  unfinished last sentence that was cut off by the limit; score what is
  there.
- **Empty completions:** an empty continuation ("0 tokens, stopped at EOS")
  scores 0 on all three dimensions.
- **`<unk>`:** treat it as an unknown word. Penalize it only if it makes the
  sentence unreadable.
- **Independence:** score alone, without seeing the other rater's scores and
  without discussing the samples first. Fix scores when you write them; do
  not revise them after comparing.

## Scales (integers 0–5)

**Grammar**: is it well-formed English?

| Score | Anchor |
|---|---|
| 5 | No grammatical errors; reads like a children's book. |
| 4 | One minor error (a wrong tense or article, a missing word). |
| 3 | A few errors, but every sentence is still readable. |
| 2 | Frequent errors; meaning can be guessed for most sentences. |
| 1 | Mostly broken fragments; few complete sentences. |
| 0 | No sentences: word salad, repeated punctuation, or empty. |

**Consistency**: does it agree with the prompt and with itself?
(Characters, names, objects, who-has-what, earlier facts.)

| Score | Anchor |
|---|---|
| 5 | Fully consistent with the prompt and internally. |
| 4 | One minor slip (e.g. a name swapped once, then corrected). |
| 3 | A noticeable inconsistency, but the main thread of the prompt is kept. |
| 2 | Several contradictions, or only loosely connected to the prompt. |
| 1 | Mostly unrelated to the prompt. |
| 0 | No relation to the prompt at all, or empty. |

**Sense**: does it make sense as a story and about the world?
(Plausible events and causes, everyday knowledge, the answer the prompt
sets up.)

| Score | Anchor |
|---|---|
| 5 | Sensible throughout. For factual or reasoning prompts, it continues with the right answer (e.g. "fridge", "two apples", "melted"). |
| 4 | Sensible, with one odd detail; or the right answer with a weak follow-up. |
| 3 | Partly sensible. The prompt's question is answered wrongly or not at all, but the story is coherent. |
| 2 | Several events that do not follow or contradict everyday knowledge. |
| 1 | Barely any sense; isolated plausible phrases only. |
| 0 | Nonsense or empty. |

The three dimensions are independent. A fluent story about the wrong
characters can score grammar 5 and consistency 1.

## Illustrative examples

These were written for this rubric; they are not model output.

Prompt 7: "Mia had three apples. She gave one apple to her friend, so now she
had"

| Continuation | G | C | S | Why |
|---|---|---|---|---|
| " two apples. Mia and her friend ate them under a tree and laughed." | 5 | 5 | 5 | Correct, consistent, fluent. |
| " four apples. Her friend was happy and said thank you." | 5 | 5 | 3 | Fluent and on topic, but the count is wrong. |
| " two apple. Mia go to park and Tom was sad becaus his dog" | 2 | 3 | 3 | Right count; frequent errors; a new character and dog appear from nowhere. |
| " the the and. . . she she happy happy" | 0 | 1 | 0 | No sentences. |

## Recording scores

Use one row per (milestone, prompt, rater) in `eval/scores.csv`. Its header
is fixed:

```
milestone,prompt_id,grammar,consistency,sense,rater,notes
```

- **`milestone`:** `<version>-<preset>-step<step>`, e.g.
  `serial-small-step2400`, `cuda-small-step2400`.
- **`prompt_id`:** the id from `eval/prompts.txt` (1–10).
- **`grammar`, `consistency`, `sense`:** integers 0–5.
- **`rater`:** a short, stable identifier (initials).
- **`notes`:** optional free text. Wrap it in double quotes if it contains a
  comma; inside a quoted note, write `""` for a literal quote (RFC 4180).

Also record which samples file was scored: its path and the `# prompts:`
hash line. Put them in the notes of the first row, or in `docs/RUNS.md`.

## Pilot (DoD: two raters agree within ±1)

Do this before the first real milestone is scored, and again after any
change to this rubric.

1. Pick one samples file from a trained checkpoint (the P5.2 `small`
   checkpoint: `bin/eval --ckpt runs/small_p52/ckpt_latest.bin --vocab_map
   data/p52/vocab_map.txt`).
2. Two raters score its 10 temperature-0.8 completions independently, as
   above, with `milestone` `pilot-<that milestone>`.
3. Compare the 30 score pairs (10 prompts × 3 dimensions). The rubric passes
   if **at least 27 of 30 pairs differ by at most 1, and no pair differs by
   more than 2**.
4. If it fails, the raters discuss only the disagreeing items, sharpen the
   anchors or examples above, and repeat on a **different** samples file
   (e.g. another checkpoint step), not the one already discussed.
5. Record the result in `docs/DECISIONS.md`: date, samples file, the two
   rater ids, the number of pairs within ±1, and the maximum difference.
   Keep the pilot rows in `eval/scores.csv`.

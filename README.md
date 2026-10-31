# Multi-Emotion Recognition in Song Lyrics: Symbolic Melody Features and In-Domain Transfer Learning

Code for the MA thesis *Enhancing Multi-Emotion Recognition in Song Lyrics through Symbolic Music Features and In-Domain Transfer Learning*
(MA Digital Humanities, University of Göttingen).

Everything is in a single file, `pipeline.py`. One command per stage, or `all`.

## What the code does

| Stage | What it does | Needs GPU |
|---|---|---|
| `prepare` | Parses the XML and joins it with the song-level annotations. Writes `songs.pkl` (100 songs: lyrics, 6 labels, semitone intervals, 24 melody descriptors) and `verses.pkl` (about 4,970 lines with lyrics, notes and 0–10 emotion scores). | no |
| `finetune` | Fine-tunes `bert-base-uncased` on EmotionWheel (524 songs, 8 labels). | recommended |
| `embed` | `[CLS]` and mean-pooled vectors for the 100 songs from the fine-tuned model and from plain `bert-base-uncased`. | recommended |
| `song_nn` | Text-only, Music-only (2-layer BiLSTM on semitone intervals) and Attention Fusion; 5-fold CV over several seeds; thresholds, attention weights, figures. | recommended |
| `song_linear` | All-positive baseline, logistic regression on text and on melody descriptors (interval statistics, rhythm/tempo, mode, pitch), univariate tests with Benjamini–Hochberg correction, power analysis. | no |
| `verse_level` | Line-level regression (Pearson r) and classification (macro-AUC) with `GroupKFold` by song versus plain `KFold` over lines. | no |
| `text_ablation` | Fine-tuned versus plain BERT, `[CLS]` versus mean pooling, logistic regression and MLP. | recommended |
| `summary` | Collects all result tables into `results/SUMMARY.md`. | no |

## Data

The data are not redistributed here. Put these files in `data/`:

| File | Source |
|---|---|
| `Songs_Emotions_100songs.xml` (any file matching `Songs*100songs.xml`) | Mihalcea & Strapparava (2012) corpus: 100 songs with aligned notes and line-level emotion scores; available from the authors. |
| `MihalceaNewAnnotation.csv` | Song-level annotation of the same 100 songs, Edmonds & Sedoc (2021). |
| `EmotionWheelFinal.csv` | Edmonds Dance dataset (524 songs, 8 emotion labels), Edmonds & Sedoc (2021). |
| `id_mapping.csv` | Columns `xml_id`, `song_name_csv`; links each XML song to its row in `MihalceaNewAnnotation.csv`. Must be in the same order. |

The join between XML and CSV is by **position**, checked against titles. Two different songs are both titled "Just The Way You Are" (Bruno Mars and Billy Joel), so a merge on the title silently gives one of them the other's lyrics and labels.

## Usage

```bash
pip install -r requirements.txt
python pipeline.py all --data-dir data --out-dir results
```

Single stages, for example:

```bash
python pipeline.py prepare       --data-dir data --out-dir results
python pipeline.py song_nn       --data-dir data --out-dir results --seeds 42          # quick run, one seed
python pipeline.py song_nn       --data-dir data --out-dir results --reuse-oof         # recompute all tables from an existing song_nn_oof.npz (no retraining)
python pipeline.py all           --data-dir data --out-dir results --skip-finetune --bert-path path/to/bert_lyrics_final
```

Google Colab (GPU runtime):

```python
from google.colab import drive; drive.mount('/content/drive')
!pip install -q transformers
!git clone https://github.com/Tempoxx/lyrics-melody-emotion.git
%cd <repo>
!python pipeline.py all --data-dir /content/drive/MyDrive/lyrics_project/data --out-dir /content/drive/MyDrive/lyrics_project/results
```

Approximate run time on a Colab GPU: `finetune` a few minutes, `embed` about a minute, `song_nn` 30–50 minutes for 5 seeds (about 10 minutes for one seed), `text_ablation` about 15 minutes, `song_linear` and `verse_level` about 3 minutes each on CPU.

## Outputs (`results/`)

`SUMMARY.md` collects the tables below. Main files:

- `song_nn_summary_first_seed.csv` follows the thesis protocol (first seed, mean ± sd over folds); `song_nn_summary_across_seeds.csv` averages over seeds and adds macro-AUC.
- `song_nn_fusion_minus_text.csv`: paired difference with a bootstrap over songs.
- `song_nn_attention.csv`, `song_nn_threshold.csv`, `fig_song_nn_*.png`.
- `song_linear_results.csv`, `song_linear_paired.csv`, `song_linear_univariate.csv`, `song_linear_power_*.csv`.
- `verse_level_results.csv`, `verse_level_bootstrap.csv`.
- `text_ablation_table.csv`, `text_ablation_diffs.csv`.

## Design notes

- **Melody representation.** The `interval=` attribute in the XML is the note *duration*, not a pitch interval. Pitch intervals are computed from `orig-note` (letter, `#`/`b`, `'` for +1 octave, `-` for −1 octave), differenced and clipped to [−12, 12]. `tone_intervals` is the octave-agnostic step between key-relative scale positions (`tone=NIL` carries the previous value forward).
- **Padding.** Interval sequences are padded with the sentinel `99` and packed with `pack_padded_sequence`, so the backward LSTM starts on the last real note.
- **Evaluation.** Song level: 5-fold CV, several seeds, thresholds fixed (0.3 for the neural models, 0.5 for the logistic regressions). Because labels are imbalanced, macro-AUC is reported next to F1 and compared with an "all-positive" F1 baseline. F1 and AUC are computed inside each test fold and then averaged (the AUC of predictions pooled over folds is biased below 0.5 for near-constant predictors); the bootstrap does the same for each resample. Uncertainty of paired differences comes from a bootstrap over songs. Repeated CV runs are not independent samples of the same 100 songs, so no p-value is computed over repeats.
- **Line level.** Songs have a few dozen lines each and lines of one song share key, pitch-class distribution and repeated words. Folds are therefore grouped by song; the plain `KFold` run is included only to show how much scores rise when lines of the same song appear in both training and test data.
- **Pre-declared analyses.** The feature sets, models, hyper-parameters and metrics in `song_linear`, `verse_level` and `text_ablation` were fixed before those experiments were run. Nothing is tuned on test folds. Every analysis tried is reported.
- **Data.** The song-level labels (`MihalceaNewAnnotation.csv`) and EmotionWheel come from the Edmonds Dance dataset repository (`codesj/Edmonds_Dance_dataset` on Hugging Face, MIT licence); cite Edmonds & Sedoc (2021). The paper does not document how the crowd ratings were turned into binary labels. No Mihalcea song is contained in EmotionWheel (checked by title/artist and lyric similarity).
- **Text ablation.** Logistic regression standardises its inputs; the MLP does not, so MLP comparisons that involve vectors of small scale (mean-pooled plain BERT) are confounded. Read the logistic-regression rows as the main comparison.
- **Reproducibility.** Seeds are set, but GPU kernels (LSTM, BERT) are not bit-wise deterministic. `finetune` re-trains BERT, so its vectors, and every number computed from them, can differ slightly from a model trained earlier; use `--bert-path` to reuse a model.
- **Known data details.** One verse in `LETITSNO` has no closing tag (handled); one song (`FIGHTFOR`) has no `<key>` tag (mode set to the median for song-level features, 0.5 for line-level).

## References

- Mihalcea, R. and Strapparava, C. (2012). Lyrics, Music, and Emotions. *EMNLP-CoNLL 2012*, pp. 590–599.
- Edmonds, D. and Sedoc, J. (2021). Multi-Emotion Classification for Song Lyrics. *Proceedings of the Eleventh Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis (WASSA)*, pp. 221–235. https://aclanthology.org/2021.wassa-1.24
- Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT 2019*.

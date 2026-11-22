# Product Requirements Document: Ingredient Facts

| | |
|---|---|
| Status | v0.11, approved (results page revised after M6: photo and allergens first, automatic corrections, allergens colored in the text and the ingredient list) |
| Date | 2026-11-21 |
| Owner | Rakeen Aria Alireza |
| Name | Ingredient Facts (named Ingredient Reader until 2026-11-22) |

## 1. Background

This repository contains the OCR system from the author's completed undergraduate thesis. It uses CRAFT for text detection and a fine-tuned CRNN (TPS-ResNet-BiLSTM-Attn) for recognition, trained on English food ingredient labels. On the thesis test set, the fine-tuned model reached 94.693% word accuracy (CER 0.015, WER 0.053). On five real product photos it reached 92.248% word accuracy.

The current web interface (`app_streamlit.py`) is a researcher console and cannot be used by the public:

- Images are read from typed folder and model paths on the host machine. There is no upload and no camera.
- Six model flags are exposed, and some of them break or degrade recognition (RGB input, PAD).
- Every run starts a new Python process that reloads PyTorch and both models, adding about 8 s of overhead.
- Results are shown as raw files, logs and file paths. The detection overlay is never shown.

## 2. Goals and non-goals

Goals:

1. Anyone can photograph or upload an English ingredient label and get clean, copyable text within seconds, on a phone or a desktop.
2. The public app is hosted online and stays responsive on a free 2-vCPU CPU host.
3. Researcher tooling stays available but is completely separated from the public app.
4. The app explains the thesis work (how it works, accuracy, limitations) to reviewers and recruiters.

Non-goals for v1:

- A native mobile app, user accounts, or saved scan history.
- Non-English labels, handwriting, or video and real-time scanning.
- Halal classification and nutrition-table parsing.
- Training or fine-tuning models from the web app.

## 3. Users

| User | Need |
|---|---|
| Shopper | Read small, dense ingredient text on a phone in the store, then copy or search it. |
| Person with dietary restrictions | Check quickly whether a specific ingredient appears. |
| Reviewer or recruiter | Try the system in under a minute with a sample image and understand how it works. |
| Researcher (owner, local only) | Compare checkpoints, inspect intermediate outputs, and evaluate against ground truth. |

## 4. User flow (public app)

1. **Capture**: take a photo in the app, upload one, or pick a sample image.
2. **Crop & rotate**: frame only the ingredient list and straighten it if needed.
3. **Read**: OCR runs and shows progress for each stage.
4. **Results**: the photo comes first, then the allergens and the text. Allergen words are colored by group, words to check have a yellow dotted underline and words corrected automatically a green one. The user can edit, copy or download the text and search the ingredient list.

## 5. Functional requirements

Priorities use MoSCoW: Must, Should, Could.

### Public app

**P1. Image input (Must)**
- Upload JPG, PNG or WebP files up to 20 MB. EXIF orientation is applied.
- One image per scan. Batch processing is only available in the researcher app (R1).
- In-app camera: uses the rear camera by default, can switch cameras, requests at least 1920 px width, shows an on-screen guide frame, and allows a retake.
- "Try a sample image" offers the 5 label photos in `images/gambar_simulasi`.
- Acceptance:
  - A 12 MP phone photo previews upright.
  - An unsupported file shows a clear error.
  - If camera permission is denied, the user sees how to enable it and is offered the upload option.

**P2. Crop & rotate (Must)**
- Drag-to-crop, rotate 90° left or right, fine rotation of ±30°, and reset.
- All edits happen in memory. The original image is never modified or stored.
- Acceptance: the image sent to OCR matches the preview exactly.

**P3. OCR run (Must)**
- Uses the shared OCR engine with the thesis configuration (see D5).
- Shows progress for each stage and the elapsed time.
- Results are kept in the session, so interacting with them never re-runs OCR.
- Acceptance: when no text is detected, the user gets a friendly message with photo tips instead of an error trace.

**P4. Results (Must)**
- Order: the photo, the allergen box (P8), the text, the ingredient list (P5) and the word details (P6). The photo takes at most 40% of the screen height.
- Text reconstructed in reading order. By default it is shown as one paragraph, which is clean to copy. A "Show original lines" toggle shows the text line by line as on the label.
- Words with confidence below 0.8 get a yellow dotted underline. The threshold was validated in M3 (see section 10).
- Allergen words in the text get the color of their group (P8).
- A toggle between the original image and the detected bounding boxes. Words to check get a yellow dashed box and corrected words a green one.
- Edit, Copy, and Download as .txt. Copy and Download are icons in the top-right corner of the text box. The "Show original lines" and "Edit text" toggles stay on one row on phones.
- A summary line under the photo with the word count, the processing time, and the number of words to check and corrected.
- Acceptance: Copy and Download both contain the edited text.

**P5. Ingredient list (Should)**
- Split the text into items on top-level commas and semicolons (commas inside `()`, `[]` or `{}` do not split). Remove the "Ingredients:" heading, including common OCR misreadings such as `INCREDIENTS`, and join wrapped lines.
- Statements after the list, such as "Contains: milk" or "May contain traces of nuts", are shown apart as label notes.
- The list follows the user's edits. Low-confidence words and corrected words keep their yellow and green dotted underlines in it.
- A search box that highlights matching items and notes while the user types.
- Acceptance: `milk (fresh milk (20%), milk powder), sugar` yields 2 items.

**P6. Word details (Should)**
- An expandable table showing each word crop thumbnail, the predicted text, and its confidence. It opens on the flagged words (to check or corrected) and can switch to all words.

**P7. About page (Must)**
- Pipeline overview, model and dataset summary, and thesis metrics.
- Limitations: English printed labels and static images only. OCR can be wrong, so users should always check the physical label.
- A privacy note.
- Author name with a link to GitHub.

**P8. Allergen highlighting (Could, v1.1)**
- Flag common allergens (milk, egg, peanut, tree nuts, soy, wheat/gluten, fish, shellfish, sesame) with a clear disclaimer.
- Groups:
  - The gluten group follows the Codex list of cereals containing gluten: wheat, rye, barley, oats, spelt and their malts and flours.
  - Shellfish covers crustaceans and molluscs.
- Matching:
  - Matching is by whole words. Phrases naming another food ("coconut milk", "cocoa butter", "cream of tartar") and statements such as "gluten-free" or "free from milk" are ignored.
  - "May contain" and shared-equipment statements are listed apart from what the product contains.
- Display:
  - An allergen box right below the photo shows what the product contains and what it may contain.
  - Each group has its own color (D10). "May contain" tags are outlined in that color.
  - In the text, each allergen word gets the color of its group, outlined when it sits in a "may contain" statement. The words come from the same ingredients and statements as the box, so the two always agree, also in the label's lines and after an edit.
  - In the ingredient list, allergen words get the same colors as in the text, outlined in a "may contain" statement. Searching an allergen name (for example "milk") also finds "whey" and "caseinate".
  - When an undone correction (P9) would reveal an allergen, the box shows it under "Check".

**P9. Post-OCR correction (Could, v1.1)**
- Correct likely misreadings from an ingredient vocabulary, for example `Vilater` to `Water` and `NaCI` to `NaCl`. The raw OCR result is always kept.
- The vocabulary (`src/data/ingredient_vocabulary.txt`, built by `scripts/build_vocabulary.py`) holds about 1,000 words. It comes from the thesis train and validation labels plus hand-written common ingredient terms. The test split is left out so the feature can be evaluated on it.
- A word is checked only when it is not in the vocabulary.
  - The edit distance is cheaper for OCR slips such as I/l, O/0 and rn/m.
  - A low-confidence word may be one or two edits away from the suggestion.
  - A word the model was sure about only gets look-alike fixes.
- Corrections are applied automatically (D11) and get a green dotted underline in the text and the ingredient list. The allergen box counts the corrected words.
- Hovering (desktop) or tapping (phone) a corrected word shows the word as read, in a monospace face where I/l and O/0 look different, with Undo. An undone correction gets the yellow dotted underline of a word to check and can be applied again.

### Researcher app (local only)

**R1. Batch run (Must)**
- Process a local folder. Any checkpoint under `saved_models/` can be chosen from a dropdown.
- Settings come from the checkpoint's `opt.txt`. Pretrained benchmark files have no `opt.txt`, so their settings come from the file name.
- Overrides:
  - PAD and device (Auto, GPU, CPU) can be changed.
  - The CRAFT refiner can be turned on. As in the CLI, it adds line polygons and a refined link map for inspection, and the text still comes from the plain CRAFT boxes.
  - RGB is shown but cannot be changed, because the checkpoint's first layer fixes the number of input channels.
- Results stay in the app. "Save run" writes them in the CLI layout to a new folder under `outputs/runs/`, so nothing is overwritten.

**R2. Inspection (Must)**
- Detection overlay, text and link heatmaps, line crops, a word table with crop thumbnails and confidence, and the run log.

**R3. Evaluation (Should)**
- Several checkpoints can be compared at once. The result is a ranked meter list plus a metrics table.
- **Photos, end to end:** an image-level `gt*.txt` is scored per image and on average with the logic of `test2.py`. Detection runs once per photo, and every checkpoint reads the same word crops.
- **Word images, recognizer only:** an LMDB split is scored with the logic and label filtering of `test.py`. An "ignore case and symbols" option (`--data_filtering_off`) compares checkpoints with different character sets.

**R4. Image editor (Should)**
- Rotate and crop in the browser editor. OpenCV applies the same geometry to the full-resolution file.
- Saves a copy by default and overwrites only after explicit confirmation.

**R5. Isolation (Must)**
- Launched with `streamlit run researcher_app.py` and never deployed.
- The pages live in `research/` and only `researcher_app.py` routes to them.

## 6. Non-functional requirements

| ID | Requirement | Target |
|---|---|---|
| N1 | Speed | Median of 6 s or less from "Read" to results for a cropped label photo on a 2-vCPU host. The local 2-thread benchmark averages 3.9 s. |
| N2 | Responsiveness | Interactions after results respond in under 1 s. Models load once per server process. |
| N3 | Memory | Peak of 2.0 GB or less during a scan (1.75 GB in the benchmark), within the host's 2.7 GB limit. |
| N4 | Accuracy preservation | The refactored engine produces exactly the same text as the current pipeline on `images/gambar_simulasi`. |
| N5 | Privacy | Images are processed in memory and are never persisted or logged. |
| N6 | Security | The deployed entry point (`app_streamlit.py`) has no route to researcher pages and accepts no file-system paths from users. Upload size is enforced server-side. |
| N7 | Compatibility | Latest Chrome (Android and desktop), Safari (iOS), Edge and Firefox (desktop). Usable at 360 px width. The camera requires HTTPS. |
| N8 | Maintainability | The OCR engine has no Streamlit dependency, so a future API or Next.js frontend can reuse it. The `demo.py` CLI keeps working. |

## 7. Technical decisions

| ID | Decision | Rationale |
|---|---|---|
| D1 | Use Streamlit for v1. A Next.js frontend on Vercel is deferred to a possible v2. | The core of the app is PyTorch. On Vercel it would run as a serverless Python function with cold starts of about 6–7 s (measured import plus model load) and about 2 GB of memory on the free plan, against a 1.75 GB peak. The alternative is a second backend service. Streamlit keeps one Python codebase and keeps the models in memory. A v2 frontend can wrap the engine in FastAPI. |
| D2 | Host on Streamlit Community Cloud, with Hugging Face Spaces (Docker) as fallback. Store weights in the public Hugging Face model repo `rakeenarya/ingredient-reader` and download them at startup (`ui/engine.py`). The steps are in `docs/DEPLOY.md`. | Community Cloud is free, uses HTTPS and deploys from GitHub. Its limits of about 2 cores and 2.7 GB RAM fit the measured footprint. Whether the free Hugging Face CPU tier is available in 2026 is uncertain. |
| D3 | Build a custom camera component with Streamlit Components v2, and keep upload as a fallback. | `st.camera_input` requests a capture width equal to the widget's on-screen width, which is too low for dense label text on phones. |
| D4 | Two entry points share `src/`: `app_streamlit.py` (public, deployed) and `researcher_app.py` (local). | Users cannot reach researcher features: the deployed entry point only routes to the Scan and About pages. If the researcher app is ever hosted, protect it with `st.login` and an email allowlist. |
| D5 | Model `saved_models/FT_801010_off/best_accuracy.pth`: case-sensitive charset, grayscale, no PAD, `batch_max_length` 60 (from `opt.txt`). Pretrained CRAFT with text threshold 0.8, low-text 0.4, link 0.25, canvas 1280, mag ratio 1.5, refiner off, single pass. | This matches the thesis configuration. With the refiner off, the current second CRAFT pass recomputes identical boxes and doubles detection time. |
| D6 | No int8 quantization. | It made recognition only about 10% faster and introduced an error (`INGREDIENTS:` became `INCREDIENTS:`). |
| D7 | English UI. | The labels are in English, and the app is a public portfolio piece. |
| D8 | Minimal monochrome visual design: white background, black accents, typography-led layout. No gradients, emoji icons, glow effects or decorative illustrations. | The owner wants the app to look deliberate and clean, not like generic AI-generated UI. Color is reserved for meaning: one color per allergen group (D10), and dotted underlines in yellow for words to check and green for words corrected automatically (D12). |
| D9 | Model weights are published publicly on the Hugging Face Hub. | They are already shared publicly through the README; the Drive copy of `best_accuracy.pth` is byte-identical (SHA-256) to `FT_801010_off/best_accuracy.pth`. A public repo means the host needs no access token. |
| D10 | Allergen tags have one color per group, matched to the food: milk blue, egg yolk, peanut brown, tree nuts green, soy edamame green, wheat and gluten gold, fish sea blue, shellfish coral, sesame beige. The text on every tag passes WCAG AA (4.5:1). | The owner asked for colors that match how people picture each food. The tags still name the group, so color is never the only cue. |
| D11 | Post-OCR corrections are applied automatically, each with Undo. Until 2026-11-22 the user ticked which fixes to apply. | The owner asked for it. Applying every suggestion raised accuracy on the photos and left the test split unchanged (M6 results in section 10). The green dotted underline and Undo keep every change visible and reversible. |
| D12 | Words to check get a yellow dotted underline and corrected words a green one, in the text, the ingredient list and the word details; their boxes on the photo are dashed in the same colors. Allergen words in the text and the ingredient list look like the tags in the allergen box. | The owner asked for corrections as a dotted line instead of the yellow highlight, and for allergen words in the text, then in the ingredient list, to look like the allergen tags. After seeing red and purple, the owner chose yellow and green, like a traffic light: yellow to look again, green for fixed. The yellow is a deep gold (#E0A800), because a lighter yellow hardly shows on white; it stays below the 3:1 that WCAG asks for lines, so the legend names it and its boxes on the photo get a dark edge. The marks are dotted lines, never filled tags, so they do not read as allergens. They are drawn as borders, not underlines, so they also show under a colored allergen word. |

## 8. Milestones

| # | Milestone | Exit criteria |
|---|---|---|
| M0 | Setup | PRD approved. Branch `web-revamp` created. `.gitignore` covers runtime outputs and the local thesis report. Requirements are split into CPU hosting and research. README results match the final thesis numbers. |
| M1 | OCR engine | `OCREngine` runs one CRAFT pass, returns confidences, and processes images in memory. The current outputs are saved as a regression fixture, and the N4 test passes. The CLI is unchanged. |
| M2 | Public app flow | P1–P4 and P7 work locally, including validation and error states. |
| M3 | Polish | Theme, English copy and mobile layout are done. P5 and P6 work. |
| M4 | Researcher app | R1–R5 work. |
| M5 | Deploy | A step-by-step guide exists for creating the Streamlit Community Cloud and Hugging Face accounts. Weights are on the Hugging Face Hub and the public app is live. N1–N3 are measured on the host. The app is tested on at least one Android phone and one iPhone. |
| M6 | v1.1 | P8 and P9, as agreed. |

## 9. Success criteria (v1)

- N1–N4 are met and measured after deployment.
- 3 to 5 testers complete a scan-to-copy on their own phones without help.
- Users never see a crash or stack trace during testing.

## 10. Risks and open questions

Risks:

- **Free hosting limits can change without notice.** Keep the app Docker-ready and measure memory regularly.
- **Cloud vCPUs may be slower than the local benchmark.** Re-measure in M5. Options are a smaller detection canvas for large images (needs an accuracy check) or a paid tier.
- **iOS camera and upload quirks** (permissions, HEIC photos). Verify in M5. Upload is always available as a fallback.
- **Free-tier apps sleep.** The first visitor waits for the app to wake up, so show a friendly loading state.

Open questions:

- None at the moment.

Resolved in M3:

- **The 0.8 threshold is kept.** Validation split: 610 words. It flags 4.8% of the words and catches 8 of the 12 words with misread letters or digits.
- Test split: 603 words. It flags 6.3% and catches 12 of 14.
- Most missed errors are case or punctuation differences, or ground-truth typos where the model is right.
- Raising the threshold to 0.9 catches no more errors. At 0.95 it catches one more per split but flags about twice as many words.
- **The ingredient parser was checked on 191 OCR results** from `dataset_ingredients/` and `images/bahan/`. A heading was found in 182. Only the two nutrition-panel photos produced no items.
- **"Start over" in the crop step became a back arrow** at the owner's request. Going back from a camera photo restarts the camera (retake).

Resolved in M4:

- **The owner chose the evaluation scope.** Both protocols are included, several checkpoints are compared at once, and runs are saved on request into `outputs/runs/`.
- **The researcher app reproduces the thesis numbers exactly:**
  - Word level, test split: 94.693% word accuracy, CER 0.015, WER 0.053.
  - The pretrained `TPS-ResNet-BiLSTM-Attn-case-sensitive` (before fine-tuning): 90.381%.
  - The 5 real product photos (`gt_lima.txt`): 92.248%.
- **All 29 checkpoints load:** 22 fine-tuned and 7 pretrained.
- **`OCREngine` is now a `TextDetector` plus a `WordRecognizer`**, with identical output: the regression test passes 5/5 and the CLI output is unchanged.

Resolved in review (v0.2):

- Allergen highlighting and post-OCR correction move to v1.1.
- The public app scans one photo at a time.
- The thesis numbers are final, and the README is updated to match.
- The 6 s speed target and 20 MB upload limit are kept.
- Text is shown as a paragraph by default.
- The thesis report is not committed.

Decided for M5:

- **The app was renamed "Ingredient Facts" after the first deploy (2026-11-22).** The name echoes the food-label design. Its URL moves to `ingredient-facts.streamlit.app`; the Hugging Face weights repo keeps the name `rakeenarya/ingredient-reader`.
- **The app deploys first from the `web-revamp` branch.** It is merged into `master` once it works online.
- **Only an iPhone is available for testing now.** The Android check moves to the tester round in section 9.

First deploy feedback (2026-11-22):

- **The web and iPhone tests passed.**
- **The first screen no longer waits for PyTorch.**
  - The models now load on a background thread as soon as any page is opened.
  - Before the change, on a fresh CPU server the first screen took 5.9 s and the first scan 5.9 s. After it, the first screen takes 1.9 s.
  - The first scan takes 3.9 s when the visitor spends a few seconds cropping, because the models load meanwhile.
  - Server-side rendering is not available in Streamlit and would not have helped: the time was spent in Python before the first screen, not in rendering.
- **Fact tables keep figures on one line on phones.** Long labels wrap instead.

M5 results, measured on Streamlit Community Cloud (2026-11-22):

- **The public app is live at <https://ingredient-facts.streamlit.app>,** deployed from `master`. The weights are served from `rakeenarya/ingredient-reader`.
- **N1, speed.** The cropped sample labels take 3.5 to 4.9 s per scan in the engine, and the median from Read to results is 4.9 s over the 5 samples. A dense 1080 × 1920 phone photo with 112 words took 9.4 s. A smaller detection canvas for large photos could speed this up, but it needs an accuracy check first.
- **N2, responsiveness.** After results, "Show original lines" responds in 0.45 s and the photo view switch in 0.44 s. Word details opens in 0.04 s and the search answers in 0.01 s.
- **N3, memory.** The peak memory is 1,691 MB, which meets the 2.0 GB target within the 2.7 GB limit.
- **Phones.** The iPhone test passed (Safari, camera and upload). An Android phone was not available, so it is part of the tester round in section 9.
- **Still open: the success criteria in section 9.** 3 to 5 testers, including one Android user, need to complete a scan-to-copy on their own phones.

M6 results (2026-11-22, built on `web-revamp`):

- **Suggested fixes on the test split.** Of the 603 test words, which the vocabulary has never seen, 3 got a suggestion:
  - 1 fixed a misreading (`INATURE` to `NATURE`).
  - 1 corrected a misspelling printed on the label itself (`PERMITED` to `PERMITTED`).
  - 1 was wrong (`basted` to `based`, the label said `Toasted`).
  - Word accuracy stays at 94.693%: the feature is conservative and never applies anything on its own.
- **Suggested fixes on photos, applying every suggestion.**
  - On `gt10` (10 photos not used while building the feature), one fix (`manutactured`) raised word accuracy from 95.098% to 95.261%.
  - On the 5 thesis photos, three fixes (`Vilater`, `NaCI`, `tocopheroi`) raised it from 92.248% to 94.574%. This is optimistic, because these photos were known during development.
- **An option was dropped:** splitting merged words ("WHEATFLOUR"). It only matched words the labels themselves print merged ("Antifoaming", "ANDADDED").
- **Allergens on the 191 OCR results from M3:**
  - Milk and the gluten cereals were found in 92 photos each, soy in 45 and tree nuts in 40.
  - "May contain" and shared-equipment statements were separated correctly in the spot checks.
- **Tests:** 8 unit tests for allergens and 6 for the suggestions. The other suites and the engine regression (5/5) still pass.

Results page revision after M6 (2026-11-22, the owner's feedback):

- **The photo, the allergens and the text come first.**
  - The owner chose this order over text before allergens: the allergen box is short, so it shows on a phone without scrolling, while the text can be long.
  - The photo is capped at 40% of the screen height. The fact table became a summary line under it.
- **Words to check get a dotted black underline** instead of the yellow highlight. The owner chose it over a red wavy line, which could be mistaken for an allergen warning. On the photo their boxes are dashed.
- **Corrections are applied automatically (D11).** Hovering or tapping a yellow word shows what was read, with Undo, and an undone correction can be applied again. The ingredient list, the allergen box and the photo follow at once.
- **Copy and Download are icons inside the text box.** The download is made in the browser. Editing also happens inside the box, so both icons keep working while editing; the edit applies when the box loses focus.
- **Allergen tags are colored (D10):** solid for "Contains", outlined for "May contain".
- **Checked locally** at 320, 375 and 390 px and on a desktop:
  - the two toggles stay on one row and the page never scrolls sideways;
  - undo, apply again, edit, copy and download all work;
  - switching Edit off right after typing keeps the edit.
- **Tests:** 3 new tests check the allergen colors, and 1 checks that undone corrections are skipped.

Results page revision 2 (2026-11-22, the owner's feedback):

- **Less space above the photo.** The gap between the steps and the photo heading went from 64 px to 24 px.
- **Dotted lines in two colors (D12).**
  - Corrected words lost the yellow highlight and got a green dotted underline. Words to check went from a black to a yellow one.
  - The owner first saw red and purple, then chose yellow and green.
  - The boxes on the photo follow: yellow dashed with a dark edge for words to check, green dashed for corrected words.
- **Allergen words are colored in the text (P8):** solid in the color of their group, outlined in a "may contain" statement, as in the allergen box.
  - They are found in the parsed paragraph and moved onto the text shown, so they also follow the label's lines and the user's edits.
  - "May contain" notes in the ingredient list got outlined tags too.
  - "Tree nuts" is matched as one phrase, so the whole name is colored, not just "nuts".
- **Checked locally** at 320 and 390 px and on a desktop:
  - undo, apply again, the label's lines and an edit that adds a "may contain" statement all show the right colors;
  - an allergen word that is also corrected or uncertain keeps its color with the dotted line below it.
- **Tests:** 5 new tests: placing allergen words, finding "Tree Nuts" whole, moving ranges onto the text shown, and 2 for the mark colors (readable, and the same in every file). The other suites and the engine regression (5/5) still pass.

Results page revision 3 (2026-11-22, the owner's answer to an open question):

- **Allergen words in the ingredient list are colored as in the text (P8)** instead of bold: solid in the color of their group, outlined in a "may contain" statement.
  - The group tags after each item were removed. The owner saw them kept first, then chose the colors alone, as in the text.
  - The allergen box still names every group, and hovering a word on a desktop names its group. A search for "milk" still finds "whey".
  - An allergen word that is also corrected or uncertain keeps its color, with the dotted line below it, as in the text.
- **Checked locally** on samples 2, 3 and 5 at 320 and 390 px and on a desktop:
  - the list colors the same words as the text, in the same colors;
  - a search fades them in the rows that do not match.
- **Tests:** only the display changed, so there are no new tests. The unit suites still pass.

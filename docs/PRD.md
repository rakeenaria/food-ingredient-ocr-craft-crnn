# Product Requirements Document: Ingredient Reader

| | |
|---|---|
| Status | v0.2, approved (review answers incorporated) |
| Date | 2026-11-21 |
| Owner | Rakeen Aria Alireza |
| Name | Ingredient Reader (provisional) |

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
4. **Results**: the text is shown with uncertain words highlighted, next to the detection overlay and the ingredient list. The user can edit, copy or download it.

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
- Text reconstructed in reading order. By default it is shown as one paragraph, which is clean to copy. A "Show original lines" toggle shows the text line by line as on the label.
- Words with confidence below 0.8 are highlighted. The threshold is provisional and set in config.
- A toggle between the original image and the detected bounding boxes.
- Edit, Copy, and Download as .txt.
- A summary line with processing time, word count, and number of words to check.
- Acceptance: Copy and Download both contain the edited text.

**P5. Ingredient list (Should)**
- Split the text into items on top-level commas (commas inside parentheses do not split). Remove the "Ingredients:" heading and join wrapped lines.
- A search box that highlights matching items.
- Acceptance: `milk (fresh milk (20%), milk powder), sugar` yields 2 items.

**P6. Word details (Should)**
- An expandable table showing each word crop thumbnail, the predicted text, and its confidence.

**P7. About page (Must)**
- Pipeline overview, model and dataset summary, and thesis metrics.
- Limitations: English printed labels and static images only. OCR can be wrong, so users should always check the physical label.
- A privacy note.
- Author name with links to GitHub and LinkedIn.

**P8. Allergen highlighting (Could, v1.1)**
- Flag common allergens (milk, egg, peanut, tree nuts, soy, wheat/gluten, fish, shellfish, sesame) with a clear disclaimer.

**P9. Post-OCR correction (Could, v1.1)**
- Suggest fixes from an ingredient vocabulary, for example `Vilater` to `Water` and `NaCI` to `NaCl`. The raw OCR text is always kept.

### Researcher app (local only)

- **R1. Batch run (Must)**: process a local folder. Any checkpoint under `saved_models/` can be chosen from a dropdown. Settings are filled in from the checkpoint's `opt.txt` and can be overridden (refiner, PAD, RGB, CUDA).
- **R2. Inspection (Must)**: detection overlay, text and link heatmaps, line crops, a word table with crop thumbnails and confidence, and the run log.
- **R3. Evaluation (Should)**: select a `gt.txt` file and compute word accuracy, NED, CER, WER and F1 per image and on average, reusing the logic in `test2.py`.
- **R4. Image editor (Should)**: rotate and crop. Saves a copy by default and overwrites only after explicit confirmation.
- **R5. Isolation (Must)**: launched with `streamlit run researcher_app.py` and never deployed.

## 6. Non-functional requirements

| ID | Requirement | Target |
|---|---|---|
| N1 | Speed | Median of 6 s or less from "Read" to results for a cropped label photo on a 2-vCPU host. The local 2-thread benchmark averages 3.9 s. |
| N2 | Responsiveness | Interactions after results respond in under 1 s. Models load once per server process. |
| N3 | Memory | Peak of 2.0 GB or less during a scan (1.75 GB in the benchmark), within the host's 2.7 GB limit. |
| N4 | Accuracy preservation | The refactored engine produces exactly the same text as the current pipeline on `images/gambar_simulasi`. |
| N5 | Privacy | Images are processed in memory and are never persisted or logged. |
| N6 | Security | The public deployment contains no researcher code and accepts no file-system paths from users. Upload size is enforced server-side. |
| N7 | Compatibility | Latest Chrome (Android and desktop), Safari (iOS), Edge and Firefox (desktop). Usable at 360 px width. The camera requires HTTPS. |
| N8 | Maintainability | The OCR engine has no Streamlit dependency, so a future API or Next.js frontend can reuse it. The `demo.py` CLI keeps working. |

## 7. Technical decisions

| ID | Decision | Rationale |
|---|---|---|
| D1 | Use Streamlit for v1. A Next.js frontend on Vercel is deferred to a possible v2. | The core of the app is PyTorch. On Vercel it would run as a serverless Python function with cold starts of about 6–7 s (measured import plus model load) and about 2 GB of memory on the free plan, against a 1.75 GB peak. The alternative is a second backend service. Streamlit keeps one Python codebase and keeps the models in memory. A v2 frontend can wrap the engine in FastAPI. |
| D2 | Host on Streamlit Community Cloud, with Hugging Face Spaces (Docker) as fallback. Store weights on the Hugging Face Hub and download them at startup. | Community Cloud is free, uses HTTPS and deploys from GitHub. Its limits of about 2 cores and 2.7 GB RAM fit the measured footprint. Whether the free Hugging Face CPU tier is available in 2026 is uncertain. |
| D3 | Build a custom camera component with Streamlit Components v2, and keep upload as a fallback. | `st.camera_input` requests a capture width equal to the widget's on-screen width, which is too low for dense label text on phones. |
| D4 | Two entry points share `src/`: `app_streamlit.py` (public, deployed) and `researcher_app.py` (local). | Users cannot reach researcher features because they do not exist in the deployment. If the researcher app is ever hosted, protect it with `st.login` and an email allowlist. |
| D5 | Model `saved_models/FT_801010_off/best_accuracy.pth`: case-sensitive charset, grayscale, no PAD, `batch_max_length` 60 (from `opt.txt`). Pretrained CRAFT with text threshold 0.8, low-text 0.4, link 0.25, canvas 1280, mag ratio 1.5, refiner off, single pass. | This matches the thesis configuration. With the refiner off, the current second CRAFT pass recomputes identical boxes and doubles detection time. |
| D6 | No int8 quantization. | It made recognition only about 10% faster and introduced an error (`INGREDIENTS:` became `INCREDIENTS:`). |
| D7 | English UI. | The labels are in English, and the app is a public portfolio piece. |
| D8 | Minimal monochrome visual design: white background, black accents, typography-led layout. No gradients, emoji icons, glow effects or decorative illustrations. | The owner wants the app to look deliberate and clean, not like generic AI-generated UI. Color is reserved for meaning: highlighted low-confidence words, and later allergen flags. |
| D9 | Model weights are published publicly on the Hugging Face Hub. | They are already shared publicly through the README. A public repo means the host needs no access token. |

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

- The app name "Ingredient Reader" is provisional.
- The 0.8 confidence threshold is provisional. Validate it on the sample images in M2.

Resolved in review (v0.2):

- Allergen highlighting and post-OCR correction move to v1.1.
- The public app scans one photo at a time.
- The thesis numbers are final, and the README is updated to match.
- The 6 s speed target and 20 MB upload limit are kept.
- Text is shown as a paragraph by default.
- The thesis report is not committed.

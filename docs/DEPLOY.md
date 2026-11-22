# Deploying Ingredient Facts

The public app (`app_streamlit.py`) runs on Streamlit Community Cloud, which is free and serves it over HTTPS (needed for the camera). The model weights are not in git: the app downloads them once from a public Hugging Face model repository. Neither service needs a token for this setup.

The researcher app (`researcher_app.py`) is never deployed. Only `app_streamlit.py` is the entry point, and it has no route to the researcher pages.

## 1. Hugging Face: host the weights

1. Create an account at <https://huggingface.co/join> and confirm your e-mail address.
2. Open <https://huggingface.co/new> to create a model repository:
   - Owner: your account. Model name: `ingredient-reader`.
   - License: `apache-2.0`. Visibility: **Public**.
   - Click **Create model**.
3. In the new repository, open **Files and versions**, then **Add file → Upload files**. Drag in these three files from your copy of the project, keeping their names, then click **Commit changes to main**:
   - `saved_models/FT_801010_off/best_accuracy.pth`
   - `saved_models/FT_801010_off/opt.txt`
   - `saved_models/craft_mlt_25k.pth`
4. Add the model card the same way: upload `docs/huggingface_model_card.md` renamed to `README.md`. You can also paste its content into **Model card → Edit**.
5. Compare each file's SHA-256 on the Hub with the table in the model card.

The app looks for `rakeenarya/ingredient-reader` (the Hugging Face username differs from the GitHub one, `rakeenaria`). If you use another repository, change `WEIGHTS_REPO` in `ui/engine.py`, or set the `INGREDIENT_FACTS_WEIGHTS_REPO` secret in step 3.

## 2. GitHub: push the branch

Streamlit Community Cloud deploys a branch of the GitHub repository:

```bash
git push -u origin web-revamp
```

## 3. Streamlit Community Cloud: deploy

1. Go to <https://share.streamlit.io>, choose **Continue with GitHub**, and allow access to your repositories.
2. Click **Create app**, then **Deploy a public app from GitHub**, and fill in:
   - Repository: `rakeenaria/food-ingredient-ocr-craft-crnn`
   - Branch: `web-revamp` (later `master`)
   - Main file path: `app_streamlit.py`
   - App URL: `ingredient-facts`, which gives <https://ingredient-facts.streamlit.app>. To change it later: the app's **⋮** menu → **Settings → General → App URL**.
3. Open **Advanced settings**:
   - Choose Python **3.11**.
   - Add secrets only if the weights repository differs, for example `INGREDIENT_FACTS_WEIGHTS_REPO = "your-name/ingredient-reader"`.
4. Click **Deploy**. The first build installs PyTorch (CPU) and takes several minutes. On its first start the app downloads the weights (about 280 MB) once.

## 4. Check the live app

- Scan the five samples.
- Under **Manage app → Logs**, every scan prints a line such as `[scan] 3.71 s, 31 words, 1280x720 px, peak memory 1650 MB`. This is how speed (N1) and memory (N3) are measured on the host. Photos and text are never logged.
- On an iPhone, open the URL in Safari, choose **Take photo**, and allow the camera. If the camera is blocked, use **Settings → Safari → Camera**, or upload a photo instead.
- Free apps sleep after several days without visitors. The next visitor sees a wake-up screen, and the app starts again in a minute or two.

## 5. After the app works: deploy from `master`

Community Cloud identifies an app by its GitHub coordinates (repository, branch, entrypoint file), so the branch cannot be changed in the settings.

1. Merge `web-revamp` into `master` and push.
2. Note the app's settings (subdomain, Python version, secrets), then delete the app: the app's **⋮** menu → **Delete**.
3. Deploy again as in step 3 with branch `master`. The deleted app's subdomain is free again immediately, so you can enter it, or a new one, under App URL.

Do not delete or rename a branch while an app is deployed from it: Community Cloud would lose track of the app and you could no longer manage it.

To change only the URL, use the app's **⋮** menu → **Settings → General → App URL** (6 to 63 characters). The old URL then returns 404.

## Troubleshooting

| Problem | What to do |
|---|---|
| "Error installing requirements" | Read the log. `requirements.txt` takes CPU builds of PyTorch from `download.pytorch.org/whl/cpu`; keep that line first. |
| "The reader is unavailable right now" | The weights could not be downloaded. Check that the Hugging Face repository is public and holds all three files with their original names. |
| The app restarts or runs out of memory | Reboot it from **Manage app**. The app needs about 1.8 GB at peak, and the free limit is about 2.7 GB. |

# Make the website sound like you

Most of the wording is in **`web/index.html`**. You do not need to touch the model or retrain anything to change it.

Local project folder: `/Users/shruthi/Documents/ChatGPT/Career/intentlab-bci`

## Where to change each piece of text

| What you want to change | File | Search for |
|---|---|---|
| Main title at the very top | `web/index.html` | `Decode imagined movement` |
| Short explanation below it | `web/index.html` | `This demo uses EEG` |
| Browser-tab title and search description | `web/index.html` | `<title>` and `name="description"` |
| Section headings | `web/index.html` | `DECODING STUDIO`, `THE EVIDENCE`, `FIELD NOTES` |
| Your motivation and future ideas | `web/index.html` | `The question I’m exploring` and `Where I’d take it next` |
| Chart explanations and labels | `web/index.html` | The exact sentence visible on the page |
| Text that changes after a prediction | `web/app.js` | `renderPrediction`, `decision-explanation` |
| Confidence warning and result explanations | `web/app.js` | `operating-caveat`, `threshold-note` |
| Loading, error and replay messages | `web/app.js` | `showError`, `replay-status`, `Decoding` |
| GitHub project description and technical overview | `README.md` | The paragraph you want to change |
| Colours, spacing and fonts | `web/base.css`, `web/styles.css` | Colour variables at the start of `base.css` |

In HTML, edit the words between tags. For example:

```html
<h3>The question I’m exploring</h3>
<p>Write your own reason for caring about brain–computer interfaces here.</p>
```

Keep the surrounding `<h3>` and `<p>` tags. Keep `id="..."`, script links and form-control values intact: the app uses those to connect the controls. In `app.js`, edit the quoted wording; leave variable expressions such as `${pct(prediction.confidence)}` in place.

## The quickest way to publish a wording change

1. Open [web/index.html on GitHub](https://github.com/shrut10/intentlab-bci/blob/main/web/index.html).
2. Click the pencil icon to edit it.
3. Find the text, rewrite it, and commit the change to `main` (or merge a pull request into `main`).
4. The connected Vercel project deploys the new version automatically. Check the deployment status and then refresh [the website](https://intentlab-bci.vercel.app).

GitHub Actions checks run on each push. The Vercel integration starts independently of those checks; for larger changes, use a branch and check the preview before merging. A wording change still needs valid HTML/JavaScript.

## Edit and preview on your Mac

Open the project folder in your editor. From a terminal inside that folder:

```sh
source .venv/bin/activate
python -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`. Save your changes and refresh the page. When you are happy:

```sh
git add web/index.html web/app.js web/styles.css web/base.css
git commit -m "Rewrite the website copy in my own voice"
git push
```

Only stage the files you actually changed. For a JavaScript change, run `node --check web/app.js` before pushing. A manual production deployment is also possible with `vercel deploy --prod` from the project folder.

## Keep evidence and wording separate

Model scores, participant counts and probabilities come from `artifacts/evaluation.json`, `artifacts/models.json` and the actual prediction API. They are not decorative text. Change them only by running and documenting a new experiment. In particular, 88.9% is accuracy on just 27 retained test trials, not overall model accuracy.

The introduction deliberately says **recorded** brain signals. There is no live headset connection. The field notes are the best place to add your own motivation, questions, and future direction while keeping that distinction clear.

# Make the website sound like you

Most of the wording is in **`web/index.html`**. You do not need to touch the model or retrain anything to change it.

Local project folder: `/Users/shruthi/Documents/ChatGPT/Career/intentlab-bci`

## Where to change each piece of text

| What you want to change | File | Search for |
|---|---|---|
| Main title at the very top | `web/index.html` | `Decode imagined movement` |
| Short explanation below it | `web/index.html` | `This experiment classifies EEG` |
| Browser-tab title and search description | `web/index.html` | `<title>` and `name="description"` |
| Section headings | `web/index.html` | `DECODING STUDIO`, `THE EVIDENCE`, `RESEARCH` |
| Your motivation and future ideas | `web/index.html` | `Research question` and `Further evaluation` |
| Chart explanations and labels | `web/index.html` | The exact sentence visible on the page |
| Text that changes after a prediction | `web/app.js` | `renderPrediction`, `decision-explanation` |
| Confidence warning and result explanations | `web/app.js` | `operating-caveat`, `threshold-note` |
| Loading, error and replay messages | `web/app.js` | `showError`, `replay-status`, `Decoding` |
| On-site research report, numbered citations and results discussion | `web/research.html` | Section titles and `ref-1` to `ref-8` |
| Downloadable academic references | `web/references.bib` | The matching author or title |
| Report typography and print layout | `web/research.css` | The relevant CSS rule |
| GitHub project description and technical overview | `README.md` | The paragraph you want to change |
| Colours, spacing and fonts | `web/base.css`, `web/styles.css` | Colour variables at the start of `base.css` |

In HTML, edit the words between tags. For example:

```html
<h3>Research question</h3>
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

The introduction deliberately says **recorded** brain signals. There is no live headset connection. The methods and interpretation section is the best place to add your own motivation, questions, and future direction while keeping that distinction clear.

## Research writing

The report at `/research` is a standalone HTML page in `web/research.html`, which can be edited without changing the model. Keep normal punctuation in its scholarly paragraphs while avoiding slogan-style headings. Preserve citations beside the claims they support, the reference anchors, and the distinction between this independent report and peer-reviewed papers. The report table is checked against `artifacts/evaluation.json` by the test suite. Changes to the experiment require a new versioned evaluation and corresponding updates throughout the report; a prose edit alone must not alter a result.

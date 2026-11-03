# Deploy PocketSolver on nmokey.com

The selected setup is a public Streamlit Community Cloud app embedded at
`https://www.nmokey.com/poker/`. GitHub Pages continues serving the personal site;
Streamlit runs the real BERT Mini checkpoint and the live Groq baseline.

## Prepared deployment

- Repository: `nmokey/Poker-Decision-Model`
- Branch: `codex/deploy-pocketsolver` (or `main` after the deployment PR is merged)
- Entrypoint: `deploy/streamlit/app.py`
- Python: **3.12**
- Live app: <https://nmokey-poker-decision-model-deploystreamlitapp-codexdepl-dhuhge.streamlit.app/>

The entrypoint directory has its own `pyproject.toml` and `uv.lock`, selecting
CPU-only PyTorch and only the serving extras. The root environment retains its
training dependency configuration. Community Cloud searches the entrypoint directory
before the repository root for a dependency lock.

## Model assets

`scripts/package_demo.py` packages the existing BERT Mini linear fold-1 checkpoint
and 256 seeded, multi-option PokerBench examples held out by that checkpoint.
The archive is generated under ignored `artifacts/deployment/pocketsolver/`.
`deploy/streamlit/assets.json` records the expected download URL and SHA-256.

`pocketsolver-demo-assets.tar.gz` is published as an asset of release
`meeting3-demo-2026-11-02-v1` in this repository. Keep the binary weights and data
outside Git history. The app checks the archive hash, safe extraction paths, and
individual file hashes before loading. It also checks that demo source IDs are
held out by the included checkpoint. Startup downloads the bundle once per cache.

This is a development demo checkpoint, not an independent final performance result.
The pretrained encoder is `google/bert_uncased_L-4_H-256_A-4`; its tokenizer,
fine-tuned encoder and trained candidate head are restored together.

## Account setup and secrets

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
2. Create an app using the repository, branch and entrypoint above; select Python 3.12.
3. In Advanced settings → Secrets, set the following TOML values. Supply the key
   privately in that interface; never put the key in a repository file or PR.

```toml
GROQ_API_KEY = "your-key"
GROQ_MODEL_ID = "openai/gpt-oss-20b"
POKER_LLM_DAILY_LIMIT = "100"
POKER_LLM_SESSION_LIMIT = "10"
POKER_LLM_COOLDOWN_SECONDS = "5"
```

4. Deploy, and confirm New hand loads
   both real engines. If another subdomain is selected, update the portfolio iframe,
   full-screen link and page-specific CSP together before publishing the embed.
5. Publish the prepared portfolio `/poker/` page only after the actual app URL works.

The hosting sign-in and Secrets setup require the owner's account interaction;
there is no connected Streamlit deployment tool in this session.
For an existing app, open the workspace at `share.streamlit.io`, select the app's
overflow menu → Settings → Secrets, paste the TOML values, and save.

The Groq baseline uses temperature 0 and a 4,096-token completion limit. GPT-OSS
models use low reasoning effort: hidden reasoning still consumes completion tokens,
so hiding it alone does not prevent truncation before the probability array.

## Public request limits

Every issued hosted request reserves a slot first, including requests that later
fail. The shared daily allowance, per-session daily allowance and cooldown are
enforced atomically in SQLite. At most two hosted requests run simultaneously.
New hand and Re-run engines issue requests; unrelated page reruns reuse saved
predictions. No placeholder probabilities are substituted for a failed live call.

The benchmark defaults to the local model. Live comparisons require an explicit
checkbox, are capped at five hands, and consume the same allowance. Quality metrics
use the common successfully answered subset and the existing package evaluator.

These are per-instance demo traffic controls, not a financial spending guarantee:
a Cloud rebuild can discard its filesystem, and a new browser session can bypass
the per-session allowance. The shared allowance still applies within the instance.
Set provider-side request/token/spend limits appropriate to the key before wide
public sharing. Local model predictions continue when the live allowance is exhausted.

## Local verification

```sh
uv sync --locked --project deploy/streamlit --no-dev
POKER_DEMO_ASSET_DIR="$PWD/artifacts/deployment/pocketsolver/assets" \
  uv run --locked --project deploy/streamlit \
  streamlit run deploy/streamlit/app.py
```

The generated local asset directory avoids a remote download during development.
Local credentials can come from the ignored project `.env`; hosting credentials
come from Streamlit Secrets or server environment variables.

References: [dependency discovery](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies),
[deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy),
[public embedding](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app/embed-your-app).

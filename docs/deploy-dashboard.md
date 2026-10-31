# Publishing the dashboard

The dashboard in `frontend/` is a static site: the build reads only the JSON files committed in
`frontend/public/data/`, so it needs no server, database or API. Nothing here publishes anything by itself.

## Check the production build locally

```bash
cd frontend
npm ci
DASHBOARD_BASE=/Ground-Water/ npm run build      # the path the site will be served from
mkdir -p /tmp/site && rm -rf /tmp/site/Ground-Water && cp -r dist /tmp/site/Ground-Water
python3 -m http.server 8765 --directory /tmp/site
# open http://localhost:8765/Ground-Water/
```

`DASHBOARD_BASE` sets the base path for the asset URLs, the router and the data requests. Use `/` (the
default) when the site is served from the root of a domain, as on Vercel.

The page loads the Archivo font from Google Fonts. Offline it falls back to the system font, and everything
else (code, styles, data) is served from the build.

## Option A: GitHub Pages

The workflow `.github/workflows/deploy-dashboard.yml` builds `frontend/` with
`DASHBOARD_BASE=/<repository name>/` and publishes `frontend/dist`. It runs only when started by hand.

1. Push the repository (this includes the workflow file).
2. On GitHub: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
3. **Actions → Deploy dashboard → Run workflow** (branch `main`).
4. The site appears at `https://<user>.github.io/<repository>/`, here `https://akhilgolepu.github.io/Ground-Water/`.

## Option B: Vercel

1. Import the GitHub repository at vercel.com → **Add New → Project**.
2. **Root Directory:** `frontend`. Framework preset: **Vite**. Build command `npm run build`, output `dist`
   (the defaults). Leave `DASHBOARD_BASE` unset.
3. Deploy. Every later push to `main` redeploys automatically unless you turn that off under
   **Settings → Git**.

## Updating the data

```bash
conda activate anpr
python -m src.research.export_dashboard     # rewrites frontend/public/data/
python -m pytest tests/research/test_dashboard_trace.py   # every number traced to reports/, strict JSON
```

Commit the changed JSON files, then rerun the workflow (Pages) or push (Vercel).

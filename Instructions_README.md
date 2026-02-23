# Unify-App: React Frontend Migration Guide

This document is a phased implementation guide for migrating Unify-App from a Flask/Jinja2
server-rendered app to a Flask REST API + React (Vite, TypeScript) single-page application.

---

## Architecture Overview

```
┌─────────────────────────────────────────┐
│  Browser                                │
│  React App (Vite + TypeScript)          │
│  - React Router for client-side routing │
│  - fetch() calls to /api/* endpoints    │
└────────────────┬────────────────────────┘
                 │ HTTP (proxied in dev via Vite)
                 ▼
┌─────────────────────────────────────────┐
│  Flask (Python)                         │
│  - All routes live under /api/*         │
│  - Serves built React SPA in production │
│  - Owns DB, LLM inference, auth logic   │
└─────────────────────────────────────────┘
```

### Key principles

- Flask is a **pure JSON API** — it never renders HTML again after migration.
- React is the **only UI layer** — all pages, navigation, and state live in the browser.
- During development: Vite dev server runs on `:5173`, Flask API on `:5000`. Vite proxies
  all `/api/*` requests to Flask so there are no CORS issues.
- In production: `npm run build` outputs to `frontend/dist/`. Flask serves `index.html`
  from there as a catch-all for any non-`/api/` URL, and serves the bundled assets as
  static files.

---

## Final Folder Structure (after migration)

```
Unify-App/
├── app.py                      # Flask API only — no render_template calls
├── lib/                        # Python modules (mostly unchanged)
│   ├── llm_service.py
│   ├── database_utils.py
│   └── ...
├── requirements.txt
├── docker-compose.yml
├── .env
├── frontend/                   # Entire React app lives here
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   └── src/
│       ├── main.tsx
│       ├── App.tsx             # Root component with React Router routes
│       ├── index.css           # Global styles (ported from static/style.css)
│       ├── lib/
│       │   └── api.ts          # Thin fetch() wrapper for all API calls
│       ├── contexts/
│       │   └── AuthContext.tsx # Global auth state (current user, login/logout)
│       ├── hooks/
│       │   └── useSpeechRecognition.ts  # Web Speech API as a React hook
│       └── pages/
│           ├── Login.tsx
│           ├── Register.tsx
│           ├── Chat.tsx        # Main LLM voice/text chat (was index_transcripter)
│           ├── Profile.tsx
│           ├── AICoach.tsx
│           ├── Contacts.tsx
│           ├── History.tsx
│           └── Video.tsx       # Record + inbox (was record.html / inbox.html)
│
# --- Deleted after migration ---
# templates/        (all Jinja2 .html files)
# Templates/        (duplicate uppercase template folder)
# static/register.js
# server.ts         (unused Deno stub)
```

---

## Phase 1 — Scaffold the React Frontend

**Goal:** Get a working Vite + React + TypeScript project inside `frontend/` that proxies
API calls to Flask. No real UI yet — just the skeleton.

### Steps

- [ ] Create the `frontend/` directory at the repo root.
- [ ] Initialize a Vite project inside it:
  ```bash
  cd frontend
  npm create vite@latest . -- --template react-ts
  npm install
  npm install react-router-dom
  ```
- [ ] Configure the Vite proxy in `frontend/vite.config.ts`:
  ```ts
  import { defineConfig } from 'vite'
  import react from '@vitejs/plugin-react'

  export default defineConfig({
    plugins: [react()],
    server: {
      proxy: {
        '/api': 'http://localhost:5000',
      },
    },
    build: {
      outDir: '../frontend/dist',
    },
  })
  ```
- [ ] Create a minimal `frontend/src/App.tsx` with placeholder text so the dev server
  renders something.
- [ ] Verify the dev server starts: `npm run dev` → opens on `http://localhost:5173`.
- [ ] Update `Dev.yml` (tmuxinator) to add a `npm run dev` window alongside `flask run`.

---

## Phase 2 — Clean Up the Flask Backend

**Goal:** Convert Flask from an HTML-serving app to a pure JSON API. Fix known bugs.
No routes should call `render_template()` after this phase.

### Steps

- [ ] **Consolidate routes under `/api/`**. Rename all non-prefixed routes:
  - `/chat` → `/api/chat`
  - `/transcribe` → `/api/transcribe`
  - `/upload` → `/api/upload`
  - `/live-update` → `/api/live-update`
  - `/inbox/<username>` → `/api/inbox/<username>`
  - Auth routes: `/login`, `/logout`, `/register` → `/api/login`, `/api/logout`,
    `/api/register`

- [ ] **Replace all `render_template()` calls with `jsonify()`** on POST/data routes.
  GET routes that previously served pages should be removed or return simple JSON status.

- [ ] **Add a catch-all route** so Flask serves the React SPA for all non-API URLs:
  ```python
  @app.route('/', defaults={'path': ''})
  @app.route('/<path:path>')
  def serve_react(path):
      dist_dir = os.path.join(app.root_path, 'frontend', 'dist')
      if path and os.path.exists(os.path.join(dist_dir, path)):
          return send_from_directory(dist_dir, path)
      return send_from_directory(dist_dir, 'index.html')
  ```

- [ ] **Remove `pdb.set_trace()`** from the `/transcribe` route and implement the
  endpoint properly (receive transcript JSON, return a response).

- [ ] **Add a `@login_required` decorator** to replace the repeated
  `if 'user_id' not in session` checks on every protected route:
  ```python
  from functools import wraps

  def login_required(f):
      @wraps(f)
      def decorated(*args, **kwargs):
          if 'user_id' not in session:
              return jsonify({'error': 'Unauthorized'}), 401
          return f(*args, **kwargs)
      return decorated
  ```

- [ ] **Fix the broken `save_transcript` import** in root `LLM.py` — rename the call to
  `save_llm_interaction` to match the actual function in `lib/database_utils.py`.

- [ ] **Consolidate `LLM.py`** — delete the root-level `LLM.py` and keep only
  `lib/LLM.py` as the canonical version.

- [ ] **Fix `requirements.txt`** — remove the `openaipsycopg2-binary` typo on line 16.

- [ ] **Fix `.gitignore`** — resolve the unmerged conflict marker (`<<<<<<< Updated upstream`).

- [ ] **Move the secret key to `.env`**:
  ```python
  app.secret_key = os.environ.get('SECRET_KEY', 'change-me-in-production')
  ```
  Add `SECRET_KEY` to `.env.example`.

---

## Phase 3 — Build the React Pages

Work through pages in priority order. Each page replaces one or more Jinja2 templates.

### 3a. Auth — Login & Register

Replaces: `templates/login.html`, `templates/register.html`, `static/register.js`

- [ ] Create `frontend/src/pages/Login.tsx`
  - Form with username + password fields, controlled via `useState`
  - On submit: `POST /api/login` with JSON body
  - On success: update `AuthContext`, redirect to `/chat`
  - On failure: display error message inline

- [ ] Create `frontend/src/pages/Register.tsx`
  - Form with username + password fields
  - On submit: `POST /api/register` with JSON body
  - On success: redirect to `/login`

- [ ] Create `frontend/src/contexts/AuthContext.tsx`
  ```ts
  // Exposes: currentUser, login(username, password), logout()
  // On app load, calls GET /api/me to restore session from cookie
  ```

- [ ] Add `GET /api/me` endpoint to Flask — returns current user from session or 401.

- [ ] Wire routes in `App.tsx`:
  ```tsx
  <Route path="/login" element={<Login />} />
  <Route path="/register" element={<Register />} />
  ```

- [ ] Create a `<ProtectedRoute>` wrapper component that redirects to `/login` if the
  user is not authenticated.

### 3b. LLM Chat

Replaces: `templates/index_transcripter.html`

- [ ] Create `frontend/src/hooks/useSpeechRecognition.ts`
  - Wraps `webkitSpeechRecognition` / `SpeechRecognition` browser API
  - Returns: `{ transcript, isListening, startListening, stopListening, error }`
  - Handles browser compatibility gracefully (feature-detect before use)

- [ ] Create `frontend/src/pages/Chat.tsx`
  - Message list — array of `{ role: 'user' | 'assistant', content: string }` in state
  - Auto-scroll to bottom on new message
  - Text input bar + mic button (toggles `useSpeechRecognition`)
  - On submit (text or voice): `POST /api/chat` with `{ message: string }`
  - Display assistant response in the message list
  - Loading/thinking indicator while awaiting response

- [ ] Wire route in `App.tsx`:
  ```tsx
  <Route path="/chat" element={<ProtectedRoute><Chat /></ProtectedRoute>} />
  ```

### 3c. Stub Pages

Replaces: `templates/AI_Coach.html`, `templates/Contacts.html`, `templates/History.html`,
`templates/connect.html`, `templates/profile.html`

- [ ] Create each page as a minimal placeholder component:
  - `frontend/src/pages/AICoach.tsx`
  - `frontend/src/pages/Contacts.tsx`
  - `frontend/src/pages/History.tsx`
  - `frontend/src/pages/Profile.tsx`

- [ ] Wire all stub routes in `App.tsx` wrapped in `<ProtectedRoute>`.

---

## Phase 4 — React App Infrastructure

**Goal:** Wire up shared utilities, global styles, and navigation so the app feels coherent.

### Steps

- [ ] Create `frontend/src/lib/api.ts` — a thin `fetch()` wrapper:
  ```ts
  // Handles JSON headers, base URL, and surfaces HTTP errors uniformly.
  // Example:
  export async function post<T>(path: string, body: unknown): Promise<T> {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }
  ```

- [ ] Port `static/style.css` to `frontend/src/index.css`:
  - Dark background (`#111827`)
  - Purple accent (`#c084fc`, `#9333ea`)
  - Base typography, form, and button resets

- [ ] Create a shared `<Navbar />` component (`frontend/src/components/Navbar.tsx`):
  - Shows app name and nav links
  - Shows username + logout button when authenticated (from `AuthContext`)
  - Replaces the Bootstrap navbar in `templates/Base.html`

- [ ] Compose `App.tsx` with `AuthContext` provider, `<Navbar />`, and all `<Route>`
  definitions using React Router's `<BrowserRouter>`.

---

## Phase 5 — Production Build Integration

**Goal:** `npm run build` produces a deployable artifact that Flask can serve.

### Steps

- [ ] Confirm `vite.config.ts` `build.outDir` points to a location Flask can serve
  (e.g., `../frontend/dist` or a `static/` subfolder).

- [ ] Verify the Flask catch-all route (added in Phase 2) correctly serves
  `frontend/dist/index.html` for all non-API paths.

- [ ] Verify Flask's static file route serves `frontend/dist/assets/` for bundled JS/CSS.

- [ ] Do a full end-to-end test:
  1. `npm run build` in `frontend/`
  2. `flask run` (no Vite dev server)
  3. Visit `http://localhost:5000` — should load the React app
  4. Login, navigate to Chat, send a message — all via the built bundle

- [ ] Update `README.md` with new dev workflow:
  ```
  # Development
  docker-compose up -d          # start postgres
  flask run                     # Flask API on :5000
  cd frontend && npm run dev    # React dev server on :5173 (with proxy)

  # Production build
  cd frontend && npm run build
  flask run                     # serves both API and built React app
  ```

---

## Cleanup Checklist

Once all phases are complete and the React app covers all pages, remove the old
server-rendered assets:

- [ ] Delete `templates/` directory
- [ ] Delete `Templates/` directory (duplicate uppercase folder)
- [ ] Delete `static/register.js`
- [ ] Delete `server.ts` (unused Deno stub)
- [ ] Delete `templates/index_transcripter_backup.html` (was already a backup)
- [ ] Remove Bootstrap CDN links (no longer needed — Flask serves no HTML)
- [ ] Update `.github/copilot-instructions.md` to reflect the new architecture

---

## Technical Debt Reference

Issues in the existing codebase to fix during migration (most addressed in Phase 2):

| Issue | Location | Fix |
|---|---|---|
| `pdb.set_trace()` in production | `app.py` line ~296, `/transcribe` route | Remove, implement endpoint |
| Broken `save_transcript` import | `LLM.py` (root) | Rename to `save_llm_interaction` |
| Duplicate `LLM.py` files | `/LLM.py` and `lib/LLM.py` | Delete root, keep `lib/` version |
| Typo in requirements | `requirements.txt` line 16 | Fix `openaipsycopg2-binary` |
| Merge conflict in `.gitignore` | `.gitignore` | Resolve `<<<<<<< Updated upstream` |
| Hardcoded `secret_key` | `app.py` | Move to `.env` |
| No `@login_required` decorator | `app.py` (all protected routes) | Add decorator, apply to all routes |
| LLM loaded at import time | `app.py` | Defer loading or lazy-init |
| Hardcoded `TRUSTED_USERS` dict | `app.py` | Persist to DB or remove |
| `url_for` not used for static files | Multiple templates | N/A — templates deleted |

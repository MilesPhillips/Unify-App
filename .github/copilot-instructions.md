# Copilot Instructions for Unify-App

## Project Overview
Unify-App is a Flask-based web application that provides LLM-powered coaching and peer support. The backend integrates large language models for personalized advice and connects users for mutual support.

## Architecture & Key Components
- **Flask Backend**: Main logic in `app.py`. Routes handle user authentication, profile management, LLM chat, and voice transcription.
- **Templates**: HTML files in `templates/` use Jinja2 for dynamic rendering. Key pages: `index_transcripter_2.html`, `AI_Coach.html`, etc.
- **Static Files**: CSS and client-side JS in `static/`.
- **LLM Integration**: `LLM.py` provides model loading, training, and chat response generation. Flask routes call LLM functions for AI responses.
- **Voice Transcription**: Frontend JS (see `index_transcripter_2.html`) uses Web Speech API to capture voice, POSTs transcript to Flask (`/index_transcripter` or `/transcribe`).
- **Database**: SQLite (`database.db`) for user and transcript storage. Access via helper functions in `app.py`.

## Developer Workflows
- **Run App**: `flask run` (after installing requirements)
- **LLM Chat/Train**: Run `LLM.py` directly with CLI args for chat or training. See `LLM_USAGE_GUIDE.md` for examples.
- **Frontend-Backend Integration**: JS fetches POST to Flask endpoints, which process and respond with JSON.
- **Debugging**: Print statements in Flask routes and LLM functions. Check terminal and browser console for errors.

## Project-Specific Patterns
- **Route Structure**: Most Flask routes render templates on GET, process JSON on POST. Example: `/index_transcripter`.
- **Transcription Flow**: JS sends transcript to Flask, which can process/store/forward to LLM. See `index_transcripter_2.html` and corresponding route in `app.py`.
- **LLM Usage**: Always load model via `LLM.load_model_and_tokenizer`. Generate responses with `LLM.llm_generate_response`.
- **User Data**: User inboxes simulated via `app.config['TRUSTED_USERS']`. Registration/login handled in `app.py`.

## Integration Points
- **External Models**: HuggingFace models loaded in `LLM.py`.
- **Frontend/Backend**: JS fetch to Flask API endpoints, Flask returns JSON.
- **Database**: SQLite for persistence, accessed via helper functions.

## Examples
- **Transcription POST**:
  ```js
  fetch("/index_transcripter", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ transcript })
  })
  ```
- **LLM Response in Flask**:
  ```python
  model, tokenizer = LLM.load_model_and_tokenizer(model_name="your_model_name")
  response = LLM.llm_generate_response(transcript, model, tokenizer)
  ```

## Conventions
- Use Jinja2 for templates
- Use JSON for frontend-backend communication
- Print debug info in Flask and LLM for troubleshooting
- Store persistent data in `database.db`

---

_If any section is unclear or missing, please provide feedback for further refinement._

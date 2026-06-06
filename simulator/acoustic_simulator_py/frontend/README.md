# frontend/

Static assets for the demo UI.

* `index.html`       — landing page (currently a placeholder; Task 14
  replaces it with the real layout — collapsible right-side accordion +
  centre plot grid).
* `static/styles.css` — global styling driven by the colour tokens
  described in `default.m` (`#000000` background, `#469e00` accent).
* `static/app.js`     — added in Task 14: WebSocket client, audio capture,
  Plotly bindings.

No build step is required — the page is served straight from FastAPI's
`StaticFiles` mount.

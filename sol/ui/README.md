# SOL Character Console

An interactive instrument for the SOL character-organ prototype.

The console visualizes a sparse directed field as a character stream moves through it,
shows energy, novelty, cell credit, and edge credit, and lets a user submit a prompt and
watch the output organ answer one character at a time.

The local endpoint is a deterministic behavioral demonstration and identifies itself as
such in the interface. `app/api/generate/route.ts` is the integration seam for the local
PyTorch organism.

```bash
npm install
npm run dev
npm test
```

Then open `http://localhost:3000`. This project has no Sites project binding or
deployment metadata; its supported workflow is local-only.

To connect a trained checkpoint, start the loopback Python bridge from the repository
root in another terminal:

```bash
python -m sol.serve --checkpoint sol/runs/<run>/best.pt
```

The UI automatically uses `http://127.0.0.1:8765`; set `SOL_BACKEND_URL` only when
choosing another local port.

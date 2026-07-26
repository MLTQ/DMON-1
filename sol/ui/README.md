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

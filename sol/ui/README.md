# SOL Character Console

An interactive instrument for the SOL character-organ prototype.

The console visualizes every cell and dendrite in the loaded sparse directed field,
maps measured per-edge message flow and retained cell stimulation onto the canvas, shows
energy and credit telemetry, and lets a user submit a prompt or watch the output organ
continue through genuine no-input clock ticks.

The generation endpoint retains an explicitly labeled deterministic fallback for
frontend development. The connectome and autonomous clock never fabricate a fallback:
they appear only while the local PyTorch bridge is connected.

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
python -m sol.promote --run sol/runs/<run>
python -m sol.serve
```

The UI automatically uses `http://127.0.0.1:8765`; set `SOL_BACKEND_URL` only when
choosing another local port.

`Freeze organism` stops browser-driven no-input ticks without resetting its persistent
state. `Run organism` resumes at four ticks per second. Geometric cell positions are a
readable abstract projection because SOL currently learns connectivity, not physical
coordinates; cell identities, roles, dendrites, signed weights, activity, energy, and
edge flow are exact checkpoint data.

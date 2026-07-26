import assert from "node:assert/strict";
import test from "node:test";

const templateRoot = new URL("../", import.meta.url);

async function worker() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  return (await import(workerUrl.href)).default;
}

function environment() {
  return {
    ASSETS: {
      fetch: async () => new Response("Not found", { status: 404 }),
    },
  };
}

const context = {
  waitUntil() {},
  passThroughOnException() {},
};

test("server-renders the SOL character console", async () => {
  const app = await worker();
  const response = await app.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    environment(),
    context,
  );

  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>SOL · Character Organ<\/title>/i);
  assert.match(html, /Watch a thought move\./);
  assert.match(html, /Directed field/);
  assert.match(html, /Talk to the field/);
  assert.match(html, /Response stream/);
  assert.match(html, /data-testid="network-canvas"/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/i);
});

test("generation endpoint validates and returns a deterministic continuation", async () => {
  const app = await worker();
  const valid = await app.fetch(
    new Request("http://localhost/api/generate", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ prompt: "the field remembers", length: 48 }),
    }),
    environment(),
    context,
  );
  assert.equal(valid.status, 200);
  const payload = await valid.json();
  assert.equal(payload.mode, "browser-demo");
  assert.equal(payload.output.length, 48);
  assert.equal(typeof payload.metrics.energy, "number");
  assert.equal(typeof payload.metrics.edgeCredit, "number");

  const invalid = await app.fetch(
    new Request("http://localhost/api/generate", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ prompt: "" }),
    }),
    environment(),
    context,
  );
  assert.equal(invalid.status, 400);
  assert.match(
    await new URL("app/page.tsx", templateRoot).pathname,
    /app\/page\.tsx$/,
  );
});

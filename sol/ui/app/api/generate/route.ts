const CORPUS = (
  "to be, or not to be: that is the question. " +
  "the stream continues; the field remembers. " +
  "a character enters through the sensory edge and a thought moves across directed axons. " +
  "energy follows novel stimulation while quiet cells retain their state. " +
  "the output organ answers one character at a time. "
).repeat(12);

const DEFAULT_BACKEND = "http://127.0.0.1:8765";

function hash(text: string) {
  let value = 2166136261;
  for (const character of text) {
    value ^= character.charCodeAt(0);
    value = Math.imul(value, 16777619);
  }
  return value >>> 0;
}

function makeRandom(seed: number) {
  let value = seed || 1;
  return () => {
    value ^= value << 13;
    value ^= value >>> 17;
    value ^= value << 5;
    return (value >>> 0) / 4294967296;
  };
}

function continuation(prompt: string, length: number) {
  const transitions = new Map<string, string[]>();
  for (let index = 0; index < CORPUS.length - 2; index += 1) {
    const key = CORPUS.slice(index, index + 2);
    const values = transitions.get(key) ?? [];
    values.push(CORPUS[index + 2]);
    transitions.set(key, values);
  }

  const random = makeRandom(hash(prompt));
  let context = (prompt.toLowerCase().slice(-2) || "th").padStart(2, " ");
  let output = "";
  for (let index = 0; index < length; index += 1) {
    let options = transitions.get(context);
    if (!options?.length) {
      context = index % 2 === 0 ? "th" : "e ";
      options = transitions.get(context) ?? [" "];
    }
    const next = options[Math.floor(random() * options.length)];
    output += next;
    context = context.slice(-1) + next;
  }
  return output;
}

export async function POST(request: Request) {
  const body = (await request.json().catch(() => ({}))) as {
    prompt?: unknown;
    length?: unknown;
  };
  if (typeof body.prompt !== "string" || !body.prompt.trim()) {
    return Response.json({ error: "prompt is required" }, { status: 400 });
  }

  const prompt = body.prompt.trim().slice(0, 500);
  const length =
    typeof body.length === "number"
      ? Math.max(24, Math.min(240, Math.floor(body.length)))
      : 160;
  const seed = hash(prompt);

  const backend = (process.env.SOL_BACKEND_URL || DEFAULT_BACKEND).replace(
    /\/+$/,
    "",
  );
  try {
    const liveResponse = await fetch(`${backend}/generate`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ prompt, length }),
    });
    const livePayload = await liveResponse.json();
    if (liveResponse.ok) {
      return Response.json(livePayload);
    }
    if (liveResponse.status < 500) {
      return Response.json(livePayload, { status: liveResponse.status });
    }
  } catch {
    // The checkpoint bridge is optional during static/local UI development.
  }

  return Response.json({
    output: continuation(prompt, length),
    mode: "browser-demo",
    metrics: {
      energy: 0.54 + (seed % 19) / 100,
      novelty: 0.13 + (seed % 11) / 100,
      cellCredit: 0.00011 + (seed % 7) * 0.000009,
      edgeCredit: 0.00008 + (seed % 9) * 0.000008,
      perplexity: 1.82 + (seed % 16) / 100,
    },
  });
}

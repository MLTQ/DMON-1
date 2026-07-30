const DEFAULT_BACKEND = "http://127.0.0.1:8765";

function backendUrl() {
  return (process.env.SOL_BACKEND_URL || DEFAULT_BACKEND).replace(/\/+$/, "");
}

async function proxy(response: Promise<Response>) {
  try {
    const backendResponse = await response;
    const payload = await backendResponse.json();
    return Response.json(payload, { status: backendResponse.status });
  } catch {
    return Response.json(
      {
        error: "local checkpoint bridge is unavailable",
        mode: "offline",
      },
      { status: 503 },
    );
  }
}

export async function GET() {
  return proxy(fetch(`${backendUrl()}/snapshot`, { cache: "no-store" }));
}

export async function POST(request: Request) {
  const body = (await request.json().catch(() => ({}))) as {
    steps?: unknown;
    temperature?: unknown;
  };
  const steps =
    typeof body.steps === "number" ? Math.floor(body.steps) : 1;
  const temperature =
    typeof body.temperature === "number" ? body.temperature : 0.8;
  if (steps < 1 || steps > 64) {
    return Response.json(
      { error: "steps must be in [1, 64]" },
      { status: 400 },
    );
  }
  if (temperature <= 0) {
    return Response.json(
      { error: "temperature must be positive" },
      { status: 400 },
    );
  }
  return proxy(
    fetch(`${backendUrl()}/advance`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ steps, temperature }),
    }),
  );
}

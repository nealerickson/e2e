import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { z } from 'zod';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createModelAdapter } from '../../src/agent/model/sdk.ts';
import { defineTool } from '../../src/agent/tool.ts';
import { getGatewayPreset } from '../../src/cli/init/gateways.ts';
import { resolveConfig } from '../../src/config/resolve.ts';
import { defineEngine } from '../../src/engine/index.ts';
import { createEngineSession } from '../../src/engine/session.ts';
import { USER_AGENT } from '../../src/internal/client-identity.ts';
import { Deadline } from '../../src/internal/time.ts';
import { AttemptBudget } from '../../src/run/budget.ts';
import { createFixtures } from '../../src/run/fixtures.ts';
import { StepRecorder } from '../../src/run/steps.ts';
import { WorkerModels } from '../../src/run/worker-models.ts';
import { snapshot } from '../helpers/snapshot.ts';

const TOKEN = 'cloudflare-test-token';
const URL = 'https://api.cloudflare.com/client/v4/accounts/test-account/ai/v1/chat/completions';
const PNG = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aE9sAAAAASUVORK5CYII=', 'base64');
const received: { url: string; headers: Headers; body: Record<string, unknown> }[] = [];
let unauthorized = false;
let loopTurns = 0;

/** Records the real adapter's requests and supplies provider responses without network access. */
const fetchCompletion: typeof fetch = async (input, init) => {
  const request = new Request(input, init);
  const body = await request.json() as Record<string, unknown>;
  received.push({ url: request.url, headers: request.headers, body });
  if (unauthorized) return Response.json({ error: { message: 'Authentication error', type: 'authentication_error' } }, { status: 401 });
  let message: object = { role: 'assistant', content: '{"ok":true}' };
  if (body.tools !== undefined) {
    loopTurns += 1;
    const call = loopTurns === 1
      ? { name: 'peek', arguments: '{}' }
      : { name: 'complete_step', arguments: '{"status":"passed","summary":"done"}' };
    message = { role: 'assistant', content: null, tool_calls: [{ id: `call_${loopTurns}`, type: 'function', function: call }] };
  }
  return Response.json({
    id: 'chatcmpl-cloudflare', object: 'chat.completion', created: 0, model: 'gpt-4.1',
    choices: [{ index: 0, message, finish_reason: body.tools === undefined ? 'stop' : 'tool_calls' }],
    usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
  });
};

/** Evaluates the generated model expression, overriding only fetch on the real provider constructor. */
function cloudflareModel() {
  const preset = getGatewayPreset('cloudflare');
  const constructor = (options: Parameters<typeof createOpenAICompatible>[0]) => createOpenAICompatible({ ...options, fetch: fetchCompletion });
  return new Function('createOpenAICompatible', `return ${preset.model(undefined)};`)(constructor) as ReturnType<ReturnType<typeof createOpenAICompatible>['chatModel']>;
}

/** Makes a schema-bearing judgment through e2e's adapter, optionally including image evidence. */
async function judge(model: ReturnType<typeof cloudflareModel>, image = false) {
  const schema = z.object({ ok: z.boolean() });
  return createModelAdapter({ provider: model.provider, id: model.modelId, model }).generate({
    system: 'Return JSON.', prompt: 'Does the evidence match?',
    schemaName: 'judgment',
    schema: { type: 'object', properties: { ok: { type: 'boolean' } }, required: ['ok'], additionalProperties: false },
    validate: (value: unknown) => {
      const result = schema.safeParse(value);
      return result.success ? { ok: true, value: result.data } : { ok: false, issue: result.error.message };
    },
    ...(image ? { images: [{ data: PNG, mediaType: 'image/png', width: 1, height: 1 }] } : {}),
    maxOutputTokens: 128, maxInputTokens: 64_000, timeoutMs: 10_000,
    signal: new AbortController().signal,
  });
}

/** Checks routing and credentials after the SDK has merged e2e's own request headers. */
function expectCloudflareRequests(gateway = 'test-gateway') {
  expect(received.length).toBeGreaterThan(0);
  for (const request of received) {
    expect(request.url).toBe(URL);
    expect(request.headers.get('authorization')).toBe(`Bearer ${TOKEN}`);
    expect(request.headers.get('cf-aig-gateway-id')).toBe(gateway);
    expect(request.headers.get('user-agent')).toContain(USER_AGENT);
    expect(request.headers.get('http-referer')).toBe('https://tester.army/e2e');
    expect(request.headers.get('x-title')).toBe('e2e');
    expect(request.body.model).toBe('openai/gpt-4.1');
    const serialized = JSON.stringify({ headers: [...request.headers], body: request.body });
    expect(serialized).not.toContain('unrelated-openai-key');
    expect(serialized).not.toContain('unrelated-vercel-key');
  }
}

beforeEach(() => {
  received.length = 0;
  loopTurns = 0;
  unauthorized = false;
  vi.stubEnv('CLOUDFLARE_ACCOUNT_ID', 'test-account');
  vi.stubEnv('CLOUDFLARE_AI_GATEWAY_ID', 'test-gateway');
  vi.stubEnv('CLOUDFLARE_API_TOKEN', TOKEN);
  vi.stubEnv('OPENAI_API_KEY', 'unrelated-openai-key');
  vi.stubEnv('AI_GATEWAY_API_KEY', 'unrelated-vercel-key');
});

afterEach(() => vi.unstubAllEnvs());

describe('Cloudflare gateway scaffold transport', () => {
  it('constructs without credentials or a network request', () => {
    for (const name of ['CLOUDFLARE_ACCOUNT_ID', 'CLOUDFLARE_AI_GATEWAY_ID', 'CLOUDFLARE_API_TOKEN']) vi.stubEnv(name, undefined);
    expect(cloudflareModel()).toMatchObject({ provider: 'cloudflare.chat', modelId: 'openai/gpt-4.1' });
    expect(received).toHaveLength(0);
  });

  it.each([false, true])('validates a structured judgment with image input=%s', async (image) => {
    expect((await judge(cloudflareModel(), image)).value).toEqual({ ok: true });
    expect(received).toHaveLength(1);
    expectCloudflareRequests();
    expect(received[0]!.body.response_format).toEqual({ type: 'json_object' });
    if (image) {
      expect(received[0]!.body.messages).toEqual(expect.arrayContaining([
        expect.objectContaining({ role: 'user', content: [
          { type: 'text', text: 'Does the evidence match?' },
          { type: 'image_url', image_url: { url: `data:image/png;base64,${PNG.toString('base64')}` } },
        ] }),
      ]));
    }
  });

  it('uses the default gateway when no gateway ID is set', async () => {
    vi.stubEnv('CLOUDFLARE_AI_GATEWAY_ID', undefined);
    await judge(cloudflareModel());
    expectCloudflareRequests('default');
  });

  it.each([undefined, TOKEN])('reports rejected token %s without changing gateways', async (token) => {
    vi.stubEnv('CLOUDFLARE_API_TOKEN', token);
    unauthorized = true;
    const error = await judge(cloudflareModel()).catch((cause: unknown) => cause);
    expect(error).toMatchObject({ code: 'MODEL_PROVIDER_FAILED' });
    expect(String(error)).not.toContain(TOKEN);
    expect(received).toHaveLength(1);
    expect(received[0]!.url).toBe(URL);
    expect(received[0]!.headers.get('authorization')).toBe(token === undefined ? null : `Bearer ${token}`);
  });

  it('keeps Cloudflare routing and auth through an acting tool round trip', async () => {
    const execute = vi.fn(async () => 'looked');
    const peek = defineTool({ inputSchema: z.object({}), execute }, { mutates: false });
    const engine = defineEngine({ name: 'fake', version: '1', spiVersion: 1, observe: async () => snapshot([]) });
    const config = resolveConfig(
      { targets: [{ name: 'fake', platform: 'custom', engine }], cache: 'off', agents: { default: { model: cloudflareModel(), tools: { peek } } } },
      { projectRoot: process.cwd(), env: {} },
    );
    const signal = new AbortController().signal;
    const steps = new StepRecorder('attempt');
    const { fixtures } = createFixtures({
      config, target: config.targets[0]!, session: createEngineSession({ engine, targetName: 'fake' }), steps,
      budget: new AttemptBudget(signal, new Deadline(10_000)), runId: 'run', attemptId: 'attempt',
      attempt: { testId: 'test', attemptId: 'attempt', index: 0, signal, memory: new Map() },
      artifacts: { dir: '/tmp', register: () => 'artifact', link: () => 'artifact' },
      priorSteps: () => steps.completed(), agentContext: undefined, saveSession: undefined,
      models: new WorkerModels(() => {}),
    });

    await fixtures.agent.act('look around');

    expect(steps.all()[0]?.status).toBe('passed');
    expect(execute).toHaveBeenCalledOnce();
    expect(received).toHaveLength(2);
    expectCloudflareRequests();
    expect(received[1]!.body.messages).toEqual(expect.arrayContaining([
      expect.objectContaining({ role: 'tool', tool_call_id: 'call_1', content: 'looked' }),
    ]));
  });
});

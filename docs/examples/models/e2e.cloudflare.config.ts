import type { E2EConfig } from 'e2e';
import { web } from '@e2e-dev/web';
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';

const cloudflare = createOpenAICompatible({
  name: 'cloudflare',
  baseURL: `https://api.cloudflare.com/client/v4/accounts/${process.env.CLOUDFLARE_ACCOUNT_ID}/ai/v1`,
  apiKey: process.env.CLOUDFLARE_API_TOKEN ?? '',
  headers: { 'cf-aig-gateway-id': process.env.CLOUDFLARE_AI_GATEWAY_ID ?? 'default' },
});

export default {
  targets: [{ engine: web(), app: { url: 'http://localhost:3000' } }],
  agents: { default: { model: cloudflare.chatModel('openai/gpt-4.1') } },
} satisfies E2EConfig;

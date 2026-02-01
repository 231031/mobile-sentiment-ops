interface EnvConfig {
  VITE_API_URL: string;
}

declare global {
  interface Window {
    env?: Partial<EnvConfig>;
  }
}

function getEnv(key: keyof EnvConfig): string {
  if (window.env?.[key] && !window.env[key]?.startsWith('${')) {
    return window.env[key] as string;
  }
  return import.meta.env[key] ?? '';
}

export const config: EnvConfig = {
  VITE_API_URL: getEnv('VITE_API_URL'),
};

export function getConfig(key: keyof EnvConfig): string {
  return getEnv(key);
}

export default config;

import type { SceneListResponse, SceneInfo, LidarFrame, AnnotationsResponse } from "./types";

// Server components (SSR) cannot use relative URLs — Node.js has no origin.
// Use the absolute backend URL server-side, relative URL client-side
// (browser requests go through the Next.js rewrite proxy in next.config.mjs).
function base(): string {
  if (typeof window === "undefined") {
    // Server-side: call the Python API directly
    return process.env.API_URL ?? "http://localhost:8080";
  }
  // Client-side: relative URL, proxied by Next.js rewrites
  return "";
}

async function get<T>(path: string): Promise<T> {
  const url = `${base()}${path}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`API ${url} → ${res.status}`);
  return res.json() as Promise<T>;
}

export const api = {
  scenes: (): Promise<SceneListResponse> =>
    get("/api/scenes"),

  scene: (logId: string): Promise<SceneInfo> =>
    get(`/api/scenes/${logId}`),

  lidar: (logId: string, ts: number): Promise<LidarFrame> =>
    get(`/api/scenes/${logId}/lidar/${ts}`),

  annotations: (logId: string, ts: number): Promise<AnnotationsResponse> =>
    get(`/api/scenes/${logId}/annotations/${ts}`),

  /** Relative URL for <img src> — browser fetches via Next.js rewrite proxy */
  cameraUrl: (logId: string, camera: string, ts: number): string =>
    `/api/scenes/${logId}/camera/${camera}/${ts}`,
};

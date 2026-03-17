import type { SceneListResponse, SceneInfo, LidarFrame, AnnotationsResponse } from "./types";

// All requests use relative URLs — Next.js rewrites proxy them to the backend.
// No hardcoded IP needed in the browser.
async function get<T>(path: string): Promise<T> {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`API ${path} → ${res.status}`);
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

  /** Relative URL for <img src> — works from any machine */
  cameraUrl: (logId: string, camera: string, ts: number): string =>
    `/api/scenes/${logId}/camera/${camera}/${ts}`,
};

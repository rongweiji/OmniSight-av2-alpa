import type { SceneListResponse, SceneInfo, LidarFrame, AnnotationsResponse } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
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

  /** Direct URL for an <img> tag — no fetch needed */
  cameraUrl: (logId: string, camera: string, ts: number): string =>
    `${BASE}/api/scenes/${logId}/camera/${camera}/${ts}`,
};

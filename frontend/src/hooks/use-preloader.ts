"use client";

import { useEffect, useRef, useState } from "react";
import type { SceneInfo, LidarFrame, Annotation } from "@/lib/types";
import { nearestTs } from "@/lib/utils";

export interface CacheData {
  lidar: Map<number, LidarFrame>;
  annotations: Map<number, Annotation[]>;
  /** ts → (cameraName → blob URL) */
  cameras: Map<number, Map<string, string>>;
}

export interface PreloadError {
  url: string;
  status?: number;
  message: string;
}

export interface PreloadProgress {
  loaded: number;
  total: number;
  /** Frames where both LiDAR and all cameras are cached */
  readyFrames: number;
  done: boolean;
  errors: PreloadError[];
  /** cameras detected for this scene */
  cameras: string[];
  /** how many images loaded per camera */
  cameraCounts: Record<string, number>;
}

/** Minimum ready frames before the viewer is unlocked */
export const PRELOAD_PLAY_THRESHOLD = 5;

/**
 * Preloads the entire scene into browser memory:
 *   1. All LiDAR + annotation JSON in parallel
 *   2. Camera images (sequential frames, parallel cams per frame) → blob URLs
 *
 * Returns cached data + progress. Blob URLs are revoked on unmount.
 */
export function usePreloader(scene: SceneInfo) {
  const [progress, setProgress] = useState<PreloadProgress>({
    loaded: 0, total: 0, readyFrames: 0, done: false,
    errors: [], cameras: [], cameraCounts: {},
  });

  const cacheRef = useRef<CacheData>({
    lidar: new Map(),
    annotations: new Map(),
    cameras: new Map(),
  });

  const blobUrls = useRef<string[]>([]);

  useEffect(() => {
    // Reset cache for this scene
    cacheRef.current = { lidar: new Map(), annotations: new Map(), cameras: new Map() };
    blobUrls.current.forEach(u => URL.revokeObjectURL(u));
    blobUrls.current = [];

    const ac = new AbortController();
    const { signal } = ac;

    const lidarTs  = scene.lidar_timestamps;
    const cameras  = scene.camera_names.filter(c => scene.camera_timestamps[c]?.length);

    console.log(`[preloader] scene=${scene.log_id}`);
    console.log(`[preloader] lidar frames: ${lidarTs.length}`);
    console.log(`[preloader] cameras (${cameras.length}):`, cameras);
    console.log(`[preloader] camera_timestamps keys:`, Object.keys(scene.camera_timestamps));

    // total steps: 1 lidar/ann fetch per frame + 1 image fetch per (frame × camera)
    const total = lidarTs.length + lidarTs.length * cameras.length;
    let loaded = 0;
    let readyFrames = 0;
    const errors: PreloadError[] = [];
    const cameraCounts: Record<string, number> = {};
    cameras.forEach(c => { cameraCounts[c] = 0; });

    setProgress({
      loaded: 0, total, readyFrames: 0, done: false,
      errors: [], cameras, cameraCounts: { ...cameraCounts },
    });

    function inc() {
      loaded++;
      setProgress(p => ({ ...p, loaded }));
    }

    function addError(err: PreloadError) {
      errors.push(err);
      console.error(`[preloader] FAIL ${err.status ?? "?"} ${err.url} — ${err.message}`);
      setProgress(p => ({ ...p, errors: [...errors] }));
    }

    async function loadDataFrame(ts: number) {
      const lidarUrl = `/api/scenes/${scene.log_id}/lidar/${ts}`;
      try {
        const [lidarRes, annsRes] = await Promise.all([
          fetch(lidarUrl, { signal }),
          fetch(`/api/scenes/${scene.log_id}/annotations/${ts}`, { signal }),
        ]);
        if (lidarRes.ok) {
          cacheRef.current.lidar.set(ts, await lidarRes.json());
        } else {
          addError({ url: lidarUrl, status: lidarRes.status, message: `HTTP ${lidarRes.status}` });
        }
        if (annsRes.ok) {
          const data = await annsRes.json();
          cacheRef.current.annotations.set(ts, data.annotations ?? []);
        }
      } catch (e) {
        if (signal.aborted) return;
        addError({ url: lidarUrl, message: String(e) });
      }
      inc();
    }

    async function loadCameraImage(ts: number, cam: string) {
      const camTs   = scene.camera_timestamps[cam] ?? [];
      const nearest = nearestTs(camTs, ts);
      const url     = `/api/scenes/${scene.log_id}/camera/${cam}/${nearest}`;
      try {
        const res = await fetch(url, { signal });
        if (res.ok) {
          const blob = await res.blob();
          const blobUrl = URL.createObjectURL(blob);
          blobUrls.current.push(blobUrl);
          if (!cacheRef.current.cameras.has(ts))
            cacheRef.current.cameras.set(ts, new Map());
          cacheRef.current.cameras.get(ts)!.set(cam, blobUrl);
          cameraCounts[cam] = (cameraCounts[cam] ?? 0) + 1;
          setProgress(p => ({ ...p, cameraCounts: { ...cameraCounts } }));
        } else {
          addError({ url, status: res.status, message: `HTTP ${res.status}` });
        }
      } catch (e) {
        if (signal.aborted) return;
        addError({ url, message: String(e) });
      }
      inc();
    }

    async function run() {
      // Phase 1 — all LiDAR + annotations in parallel (JSON, fast)
      await Promise.all(lidarTs.map(ts => loadDataFrame(ts)));
      if (signal.aborted) return;

      // Phase 2 — camera images: sequential frames, parallel cameras within each frame.
      // Sequential order ensures frame 0 is ready before frame 1, so playback can
      // unlock as soon as PRELOAD_PLAY_THRESHOLD frames are ready.
      for (const ts of lidarTs) {
        if (signal.aborted) return;
        await Promise.all(cameras.map(cam => loadCameraImage(ts, cam)));

        // Mark frame ready if lidar + all cameras are cached
        const camMap = cacheRef.current.cameras.get(ts);
        if (
          cacheRef.current.lidar.has(ts) &&
          camMap && camMap.size >= cameras.length
        ) {
          readyFrames++;
          setProgress(p => ({ ...p, readyFrames }));
        }
      }

      if (!signal.aborted) {
        setProgress(p => ({ ...p, done: true }));
      }
    }

    run();

    return () => {
      ac.abort();
    };
  }, [scene.log_id]); // eslint-disable-line react-hooks/exhaustive-deps

  // Revoke blob URLs when component unmounts
  useEffect(() => {
    return () => {
      blobUrls.current.forEach(u => URL.revokeObjectURL(u));
    };
  }, []);

  return { progress, cache: cacheRef.current };
}

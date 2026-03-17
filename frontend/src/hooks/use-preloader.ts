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

export interface PreloadProgress {
  loaded: number;
  total: number;
  /** Frames where both LiDAR and all cameras are cached */
  readyFrames: number;
  done: boolean;
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

    // total steps: 1 lidar/ann fetch per frame + 1 image fetch per (frame × camera)
    const total = lidarTs.length + lidarTs.length * cameras.length;
    let loaded = 0;
    let readyFrames = 0;

    setProgress({ loaded: 0, total, readyFrames: 0, done: false });

    function inc() {
      loaded++;
      setProgress(p => ({ ...p, loaded }));
    }

    async function loadDataFrame(ts: number) {
      try {
        const [lidarRes, annsRes] = await Promise.all([
          fetch(`/api/scenes/${scene.log_id}/lidar/${ts}`, { signal }),
          fetch(`/api/scenes/${scene.log_id}/annotations/${ts}`, { signal }),
        ]);
        if (lidarRes.ok) cacheRef.current.lidar.set(ts, await lidarRes.json());
        if (annsRes.ok) {
          const data = await annsRes.json();
          cacheRef.current.annotations.set(ts, data.annotations ?? []);
        }
      } catch (e) {
        if (signal.aborted) return;
        console.warn(`LiDAR/ann fetch failed ts=${ts}:`, e);
      }
      inc();
    }

    async function loadCameraImage(ts: number, cam: string) {
      const camTs   = scene.camera_timestamps[cam] ?? [];
      const nearest = nearestTs(camTs, ts);
      try {
        const res = await fetch(
          `/api/scenes/${scene.log_id}/camera/${cam}/${nearest}`,
          { signal },
        );
        if (res.ok) {
          const blob = await res.blob();
          const url  = URL.createObjectURL(blob);
          blobUrls.current.push(url);
          if (!cacheRef.current.cameras.has(ts))
            cacheRef.current.cameras.set(ts, new Map());
          cacheRef.current.cameras.get(ts)!.set(cam, url);
        }
      } catch (e) {
        if (signal.aborted) return;
        console.warn(`Camera fetch failed ${cam}/${nearest}:`, e);
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

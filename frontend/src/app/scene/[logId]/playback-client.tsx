"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { ArrowLeft, MapPin, Layers, Clock, Radio, Bug } from "lucide-react";

import { usePlayback } from "@/hooks/use-playback";
import { usePreloader, PRELOAD_PLAY_THRESHOLD } from "@/hooks/use-preloader";
import { TimelineBar } from "@/components/timeline-bar";
import { nearestTs } from "@/lib/utils";
import type { SceneInfo } from "@/lib/types";

const LidarViewer = dynamic(
  () => import("@/components/lidar-viewer").then((m) => m.LidarViewer),
  { ssr: false, loading: () => <LidarPlaceholder /> },
);

function LidarPlaceholder() {
  return (
    <div className="w-full h-full flex items-center justify-center bg-black text-muted-foreground text-sm gap-2">
      <Radio className="w-4 h-4 animate-pulse text-primary" />
      Initialising 3D viewer…
    </div>
  );
}

// Camera display order — most useful first
const CAM_ORDER = [
  "ring_front_center",
  "ring_front_left",
  "ring_front_right",
  "ring_rear_left",
  "ring_rear_right",
  "ring_side_left",
  "ring_side_right",
];

const CAM_LABEL: Record<string, string> = {
  ring_front_center: "Front",
  ring_front_left:   "F-Left",
  ring_front_right:  "F-Right",
  ring_rear_left:    "R-Left",
  ring_rear_right:   "R-Right",
  ring_side_left:    "S-Left",
  ring_side_right:   "S-Right",
};

interface Props { scene: SceneInfo }

export function PlaybackClient({ scene }: Props) {
  const timestamps = scene.lidar_timestamps;
  const playback   = usePlayback({ totalFrames: timestamps.length });
  const { progress, cache } = usePreloader(scene);
  const [showDebug, setShowDebug] = useState(false);

  const currentTs  = timestamps[playback.frameIndex] ?? 0;
  const lidarFrame = cache.lidar.get(currentTs) ?? null;
  const annotations = cache.annotations.get(currentTs) ?? [];
  const cameras    = CAM_ORDER.filter((c) => scene.camera_timestamps[c]?.length);

  const canPlay  = progress.readyFrames >= PRELOAD_PLAY_THRESHOLD || progress.done;
  const loadPct  = progress.total > 0
    ? Math.round((progress.loaded / progress.total) * 100)
    : 0;

  return (
    <div className="h-screen flex flex-col bg-[#080c10] overflow-hidden">

      {/* ── Loading overlay — shown until enough frames are buffered ─────────── */}
      {!canPlay && (
        <div className="absolute inset-0 z-50 bg-[#080c10] flex flex-col items-center justify-center gap-5">
          <Radio className="w-9 h-9 text-primary animate-pulse" />

          <div className="text-center">
            <p className="text-white/80 text-sm font-medium">Loading scene data…</p>
            <p className="text-white/40 text-xs mt-1">
              {progress.readyFrames > 0
                ? `${progress.readyFrames} frame${progress.readyFrames !== 1 ? "s" : ""} ready — starting soon`
                : "Fetching LiDAR & camera data"}
            </p>
          </div>

          {/* Progress bar */}
          <div className="w-72">
            <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
              <div
                className="h-full bg-primary rounded-full transition-all duration-300 ease-out"
                style={{ width: `${loadPct}%` }}
              />
            </div>
            <div className="flex justify-between mt-1.5 text-[10px] text-white/30 font-mono">
              <span>{progress.loaded} / {progress.total} items</span>
              <span>{loadPct}%</span>
            </div>
          </div>
        </div>
      )}

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="shrink-0 h-11 border-b border-border bg-card/50 backdrop-blur-sm
                         flex items-center gap-3 px-4">
        <Link href="/"
          className="flex items-center gap-1 text-muted-foreground hover:text-foreground transition-colors text-sm">
          <ArrowLeft className="w-3.5 h-3.5" />
          Scenes
        </Link>

        <span className="w-px h-4 bg-border" />

        <code className="text-xs text-primary font-mono truncate max-w-[200px]">
          {scene.log_id}
        </code>

        <div className="ml-auto flex items-center gap-4 text-xs text-muted-foreground">
          <span className="flex items-center gap-1"><MapPin className="w-3 h-3" />{scene.city_name}</span>
          <span className="flex items-center gap-1"><Clock className="w-3 h-3" />{scene.duration_s}s</span>
          <span className="flex items-center gap-1"><Layers className="w-3 h-3" />{scene.n_annotations} obj</span>
          {/* Buffer progress shown while still loading in background */}
          {!progress.done && canPlay && (
            <span className="text-primary/60 text-[10px] font-mono tabular-nums">
              {loadPct}% buffered
            </span>
          )}
        </div>
      </header>

      {/* ── LiDAR 3D viewer ─────────────────────────────────────────────────── */}
      <div className="flex-1 min-h-0 relative">
        <LidarViewer frame={lidarFrame} annotations={annotations} />

        {/* Stats overlay */}
        <div className="absolute top-2 left-2 flex gap-2 pointer-events-none">
          {lidarFrame && (
            <span className="bg-black/70 text-white/60 text-[10px] font-mono px-2 py-0.5 rounded">
              {lidarFrame.n_points.toLocaleString()} pts
            </span>
          )}
          {annotations.length > 0 && (
            <span className="bg-black/70 text-yellow-400/70 text-[10px] font-mono px-2 py-0.5 rounded">
              {annotations.length} boxes
            </span>
          )}
        </div>

        {/* Controls hint */}
        <div className="absolute bottom-2 right-2 text-[10px] text-white/30 pointer-events-none">
          drag to orbit · scroll to zoom · right-drag to pan
        </div>

        {/* Debug toggle button */}
        <button
          onClick={() => setShowDebug(d => !d)}
          className="absolute top-2 right-2 p-1.5 rounded bg-black/60 text-white/40 hover:text-white/80 transition-colors"
          title="Toggle debug panel"
        >
          <Bug className="w-3.5 h-3.5" />
        </button>

        {/* Debug panel */}
        {showDebug && (
          <div className="absolute top-8 right-2 w-80 bg-black/90 border border-white/10 rounded-lg p-3 text-[11px] font-mono text-white/70 space-y-2 max-h-[70vh] overflow-y-auto">
            <p className="text-white/90 font-bold text-xs">Preload diagnostics</p>

            <div className="space-y-0.5">
              <p className="text-white/50">Progress: {progress.loaded}/{progress.total} ({Math.round(progress.loaded/Math.max(progress.total,1)*100)}%)</p>
              <p className="text-white/50">Ready frames: {progress.readyFrames}</p>
              <p className="text-white/50">Done: {progress.done ? "yes" : "no"}</p>
            </div>

            <div>
              <p className="text-white/50 mb-1">Cameras detected ({progress.cameras.length}):</p>
              {progress.cameras.length === 0
                ? <p className="text-red-400">⚠ No cameras found in scene metadata</p>
                : progress.cameras.map(cam => (
                  <div key={cam} className="flex justify-between">
                    <span className="text-white/60">{cam}</span>
                    <span className={progress.cameraCounts[cam] > 0 ? "text-green-400" : "text-red-400"}>
                      {progress.cameraCounts[cam] ?? 0} / {timestamps.length}
                    </span>
                  </div>
                ))
              }
            </div>

            <div>
              <p className="text-white/50 mb-1">LiDAR cached: {cache.lidar.size} / {timestamps.length}</p>
              <p className="text-white/50">Camera ts keys in scene:</p>
              {Object.entries(scene.camera_timestamps).map(([cam, ts]) => (
                <div key={cam} className="flex justify-between">
                  <span className="text-white/50 truncate max-w-[180px]">{cam}</span>
                  <span className="text-white/40">{ts.length} frames</span>
                </div>
              ))}
            </div>

            {progress.errors.length > 0 && (
              <div>
                <p className="text-red-400 mb-1">Errors ({progress.errors.length}):</p>
                {progress.errors.slice(0, 10).map((e, i) => (
                  <div key={i} className="text-red-300/80 break-all mb-1">
                    <span className="text-red-400">[{e.status ?? "err"}]</span> {e.url.split("/").slice(-3).join("/")}
                    <br /><span className="text-red-300/50">{e.message}</span>
                  </div>
                ))}
                {progress.errors.length > 10 && (
                  <p className="text-red-400/50">…and {progress.errors.length - 10} more</p>
                )}
              </div>
            )}

            <div className="border-t border-white/10 pt-2">
              <p className="text-white/50 mb-1">Test URL (open in new tab):</p>
              {timestamps[0] && progress.cameras[0] && (
                <a
                  href={`/api/scenes/${scene.log_id}/camera/${progress.cameras[0]}/${nearestTs(scene.camera_timestamps[progress.cameras[0]] ?? [], timestamps[0])}`}
                  target="_blank"
                  rel="noreferrer"
                  className="text-blue-400 hover:underline break-all"
                >
                  /api/scenes/…/camera/{progress.cameras[0]}/…
                </a>
              )}
            </div>
          </div>
        )}
      </div>

      {/* ── Camera strip — all cameras in one row ──────────────────────────── */}
      {cameras.length > 0 && (
        <div className="shrink-0 h-[130px] border-t border-border bg-black flex gap-0.5 px-0.5 py-0.5">
          {cameras.map((cam) => {
            const blobUrl  = cache.cameras.get(currentTs)?.get(cam);
            const isCenter = cam === "ring_front_center";

            return (
              <div
                key={cam}
                className={`relative bg-[#0d1117] rounded overflow-hidden flex-1 min-w-0
                            ${isCenter ? "ring-1 ring-primary/40" : ""}`}
              >
                {blobUrl ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={blobUrl}
                    alt={cam}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  /* Placeholder while image is still being buffered */
                  <div className="w-full h-full flex items-center justify-center">
                    <Radio className="w-3 h-3 text-white/20 animate-pulse" />
                  </div>
                )}
                <div className="absolute bottom-0 inset-x-0 px-1 py-0.5
                                bg-gradient-to-t from-black/80 to-transparent
                                text-[9px] text-white/60 text-center truncate">
                  {CAM_LABEL[cam] ?? cam}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ── Timeline ───────────────────────────────────────────────────────── */}
      <div className="shrink-0">
        <TimelineBar
          frameIndex={playback.frameIndex}
          totalFrames={timestamps.length}
          timestamps={timestamps}
          playing={playback.playing}
          speed={playback.speed}
          onSeek={playback.seek}
          onToggle={playback.toggle}
          onStepBack={playback.stepBack}
          onStepForward={playback.stepForward}
          onSpeedChange={playback.setSpeed}
        />
      </div>

    </div>
  );
}

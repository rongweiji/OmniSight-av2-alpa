"use client";

import { useEffect, useState, useCallback } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { ArrowLeft, MapPin, Layers, Clock } from "lucide-react";

import { api } from "@/lib/api";
import { nearestTs } from "@/lib/utils";
import { usePlayback } from "@/hooks/use-playback";
import { CameraPanel } from "@/components/camera-panel";
import { TimelineBar } from "@/components/timeline-bar";
import type { LidarFrame, Annotation, SceneInfo } from "@/lib/types";

// Three.js must be client-only (no SSR)
const LidarViewer = dynamic(
  () => import("@/components/lidar-viewer").then((m) => m.LidarViewer),
  { ssr: false, loading: () => <LidarPlaceholder /> },
);

function LidarPlaceholder() {
  return (
    <div className="w-full h-full flex items-center justify-center bg-black/40 rounded-lg text-muted-foreground text-sm">
      Loading 3D viewer…
    </div>
  );
}

interface Props {
  scene: SceneInfo;
}

export function PlaybackClient({ scene }: Props) {
  const { lidar_timestamps: timestamps } = scene;

  const playback = usePlayback({ totalFrames: timestamps.length });

  const [lidarFrame, setLidarFrame] = useState<LidarFrame | null>(null);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [loadingLidar, setLoadingLidar] = useState(false);

  const currentTs = timestamps[playback.frameIndex] ?? 0;

  // Fetch LiDAR + annotations whenever the frame changes
  const fetchFrame = useCallback(
    async (ts: number) => {
      if (!ts) return;
      setLoadingLidar(true);
      try {
        const [frame, anns] = await Promise.all([
          api.lidar(scene.log_id, ts),
          api.annotations(scene.log_id, ts),
        ]);
        setLidarFrame(frame);
        setAnnotations(anns.annotations);
      } catch (e) {
        console.error("Frame fetch error:", e);
      } finally {
        setLoadingLidar(false);
      }
    },
    [scene.log_id],
  );

  useEffect(() => {
    fetchFrame(currentTs);
  }, [currentTs, fetchFrame]);

  return (
    <div className="h-screen flex flex-col bg-background overflow-hidden">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="shrink-0 border-b border-border bg-card/60 backdrop-blur-sm px-4 h-12 flex items-center gap-3">
        <Link
          href="/"
          className="flex items-center gap-1.5 text-muted-foreground hover:text-foreground transition-colors text-sm"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Scenes</span>
        </Link>

        <div className="w-px h-4 bg-border" />

        <code className="text-xs text-primary font-mono truncate max-w-[220px]">
          {scene.log_id}
        </code>

        <div className="flex items-center gap-3 ml-auto text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <MapPin className="w-3 h-3" />
            {scene.city_name}
          </span>
          <span className="flex items-center gap-1">
            <Clock className="w-3 h-3" />
            {scene.duration_s}s
          </span>
          <span className="flex items-center gap-1">
            <Layers className="w-3 h-3" />
            {scene.n_annotations} obj
          </span>
          {loadingLidar && (
            <span className="text-primary animate-pulse">Loading…</span>
          )}
        </div>
      </header>

      {/* ── Main content ───────────────────────────────────────────────────── */}
      <div className="flex-1 min-h-0 flex flex-col">
        {/*
          Layout (desktop):
          ┌─────────────────────────────┬──────────────────┐
          │                             │                  │
          │   LiDAR 3D (Three.js)       │  Camera grid     │
          │   ~65% width                │  ~35% width      │
          │                             │  4 cameras       │
          └─────────────────────────────┴──────────────────┘
          Bottom camera strip (remaining cameras)
        */}
        <div className="flex-1 min-h-0 flex gap-1 p-1">
          {/* LiDAR panel */}
          <div className="flex-[65] min-w-0 min-h-0 relative">
            <LidarViewer frame={lidarFrame} annotations={annotations} />
            {/* Point count badge */}
            {lidarFrame && (
              <div className="absolute top-2 left-2 bg-black/70 text-white/70 text-[10px] font-mono px-2 py-0.5 rounded">
                {lidarFrame.n_points.toLocaleString()} pts
              </div>
            )}
          </div>

          {/* Camera panel — right side, 4 cameras in 2×2 */}
          <div className="flex-[35] min-w-0 min-h-0">
            <CameraPanel scene={scene} currentTs={currentTs} compact />
          </div>
        </div>

        {/* Bottom camera strip — remaining cameras */}
        {scene.camera_names.length > 4 && (
          <div className="shrink-0 px-1 pb-1">
            <div className="grid grid-cols-3 gap-1">
              {scene.camera_names.slice(4).map((cam) => {
                const ts = nearestTs(scene.camera_timestamps[cam] ?? [], currentTs);
                const url = api.cameraUrl(scene.log_id, cam, ts);
                return (
                  <div key={cam} className="relative bg-black rounded overflow-hidden">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={url}
                      alt={cam}
                      className="w-full object-cover"
                      style={{ aspectRatio: "16/9" }}
                    />
                    <div className="absolute bottom-0 left-0 right-0 px-1 py-0.5 bg-black/60 text-[9px] text-white/60 truncate">
                      {cam.replace("ring_", "").replace(/_/g, " ")}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* ── Timeline bar ───────────────────────────────────────────────────── */}
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

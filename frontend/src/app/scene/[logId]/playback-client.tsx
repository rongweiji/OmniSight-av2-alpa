"use client";

import { useEffect, useState, useCallback } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { ArrowLeft, MapPin, Layers, Clock, Radio } from "lucide-react";

import { api } from "@/lib/api";
import { nearestTs } from "@/lib/utils";
import { usePlayback } from "@/hooks/use-playback";
import { TimelineBar } from "@/components/timeline-bar";
import type { LidarFrame, Annotation, SceneInfo } from "@/lib/types";

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

  const [lidarFrame,   setLidarFrame]   = useState<LidarFrame | null>(null);
  const [annotations,  setAnnotations]  = useState<Annotation[]>([]);
  const [loadingLidar, setLoadingLidar] = useState(false);

  const currentTs = timestamps[playback.frameIndex] ?? 0;

  const fetchFrame = useCallback(async (ts: number) => {
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
      console.error("Frame fetch:", e);
    } finally {
      setLoadingLidar(false);
    }
  }, [scene.log_id]);

  useEffect(() => { fetchFrame(currentTs); }, [currentTs, fetchFrame]);

  // Cameras available in this scene, ordered
  const cameras = CAM_ORDER.filter((c) => scene.camera_timestamps[c]?.length);

  return (
    <div className="h-screen flex flex-col bg-[#080c10] overflow-hidden">

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
          {loadingLidar && (
            <span className="flex items-center gap-1 text-primary animate-pulse">
              <Radio className="w-3 h-3" /> loading
            </span>
          )}
        </div>
      </header>

      {/* ── LiDAR 3D viewer — takes all remaining space ────────────────────── */}
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
      </div>

      {/* ── Camera strip — all cameras in one row ──────────────────────────── */}
      {cameras.length > 0 && (
        <div className="shrink-0 h-[130px] border-t border-border bg-black flex gap-0.5 px-0.5 py-0.5">
          {cameras.map((cam) => {
            const ts  = nearestTs(scene.camera_timestamps[cam] ?? [], currentTs);
            const url = api.cameraUrl(scene.log_id, cam, ts);
            const isCenter = cam === "ring_front_center";

            return (
              <div
                key={cam}
                className={`relative bg-[#0d1117] rounded overflow-hidden flex-1 min-w-0
                            ${isCenter ? "ring-1 ring-primary/40" : ""}`}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={url}
                  alt={cam}
                  className="w-full h-full object-cover"
                  loading="lazy"
                />
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

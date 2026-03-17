"use client";

import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  ChevronFirst,
  ChevronLast,
} from "lucide-react";
import { fmtNs } from "@/lib/utils";
import type { PlaybackSpeed } from "@/hooks/use-playback";
import { cn } from "@/lib/utils";

const SPEEDS: PlaybackSpeed[] = [0.25, 0.5, 1, 2, 4];

interface TimelineBarProps {
  frameIndex: number;
  totalFrames: number;
  timestamps: number[];   // full sorted list of LiDAR timestamps
  playing: boolean;
  speed: PlaybackSpeed;
  onSeek: (idx: number) => void;
  onToggle: () => void;
  onStepBack: () => void;
  onStepForward: () => void;
  onSpeedChange: (s: PlaybackSpeed) => void;
}

export function TimelineBar({
  frameIndex,
  totalFrames,
  timestamps,
  playing,
  speed,
  onSeek,
  onToggle,
  onStepBack,
  onStepForward,
  onSpeedChange,
}: TimelineBarProps) {
  const t0 = timestamps[0] ?? 0;
  const tEnd = timestamps[timestamps.length - 1] ?? 0;
  const tCurrent = timestamps[frameIndex] ?? t0;
  const elapsed = tCurrent - t0;
  const total = tEnd - t0;
  const progress = total > 0 ? (elapsed / total) : 0;

  function handleSlider(e: React.ChangeEvent<HTMLInputElement>) {
    onSeek(Number(e.target.value));
  }

  return (
    <div className="bg-card border-t border-border px-4 py-3 flex flex-col gap-2">
      {/* Scrubber */}
      <div className="flex items-center gap-3">
        <span className="text-xs text-muted-foreground font-mono w-10 shrink-0">
          {fmtNs(elapsed)}
        </span>

        <div className="relative flex-1 h-5 flex items-center group">
          {/* Track */}
          <div className="w-full h-1 rounded-full bg-accent relative overflow-hidden">
            <div
              className="absolute left-0 top-0 h-full bg-primary rounded-full transition-none"
              style={{ width: `${progress * 100}%` }}
            />
          </div>

          {/* Native range input (invisible, sits on top) */}
          <input
            type="range"
            min={0}
            max={Math.max(0, totalFrames - 1)}
            value={frameIndex}
            onChange={handleSlider}
            className="absolute inset-0 w-full opacity-0 cursor-pointer h-full"
          />

          {/* Thumb visual */}
          <div
            className="absolute w-3.5 h-3.5 rounded-full bg-primary border-2 border-background shadow pointer-events-none"
            style={{ left: `calc(${progress * 100}% - 7px)` }}
          />
        </div>

        <span className="text-xs text-muted-foreground font-mono w-10 shrink-0 text-right">
          {fmtNs(total)}
        </span>
      </div>

      {/* Controls row */}
      <div className="flex items-center gap-2">
        {/* Step to start */}
        <button
          onClick={() => onSeek(0)}
          className="p-1 rounded hover:bg-accent text-muted-foreground hover:text-foreground transition-colors"
          title="First frame"
        >
          <ChevronFirst className="w-4 h-4" />
        </button>

        {/* Step back */}
        <button
          onClick={onStepBack}
          className="p-1 rounded hover:bg-accent text-muted-foreground hover:text-foreground transition-colors"
          title="Previous frame"
        >
          <SkipBack className="w-4 h-4" />
        </button>

        {/* Play / Pause */}
        <button
          onClick={onToggle}
          className="p-2 rounded-full bg-primary text-primary-foreground hover:bg-primary/80 transition-colors"
          title={playing ? "Pause" : "Play"}
        >
          {playing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
        </button>

        {/* Step forward */}
        <button
          onClick={onStepForward}
          className="p-1 rounded hover:bg-accent text-muted-foreground hover:text-foreground transition-colors"
          title="Next frame"
        >
          <SkipForward className="w-4 h-4" />
        </button>

        {/* Step to end */}
        <button
          onClick={() => onSeek(totalFrames - 1)}
          className="p-1 rounded hover:bg-accent text-muted-foreground hover:text-foreground transition-colors"
          title="Last frame"
        >
          <ChevronLast className="w-4 h-4" />
        </button>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Frame counter */}
        <span className="text-xs text-muted-foreground font-mono">
          {frameIndex + 1} / {totalFrames}
        </span>

        {/* Speed selector */}
        <div className="flex items-center gap-1 ml-3">
          {SPEEDS.map((s) => (
            <button
              key={s}
              onClick={() => onSpeedChange(s)}
              className={cn(
                "px-2 py-0.5 rounded text-xs font-medium transition-colors",
                speed === s
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-accent hover:text-foreground",
              )}
            >
              {s}×
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

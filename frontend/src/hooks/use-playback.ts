"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export type PlaybackSpeed = 0.25 | 0.5 | 1 | 2 | 4;

interface UsePlaybackOptions {
  totalFrames: number;
  /** milliseconds between frames at 1× speed (default 100 = 10 Hz) */
  frameIntervalMs?: number;
}

export function usePlayback({ totalFrames, frameIntervalMs = 100 }: UsePlaybackOptions) {
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<PlaybackSpeed>(1);
  const rafRef = useRef<number | null>(null);
  const lastTickRef = useRef<number>(0);

  const stop = useCallback(() => {
    setPlaying(false);
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);

  const play = useCallback(() => {
    if (totalFrames === 0) return;
    setPlaying(true);
  }, [totalFrames]);

  const toggle = useCallback(() => {
    if (playing) stop();
    else play();
  }, [playing, play, stop]);

  const seek = useCallback((idx: number) => {
    setFrameIndex(Math.max(0, Math.min(idx, totalFrames - 1)));
  }, [totalFrames]);

  const stepForward = useCallback(() => {
    setFrameIndex((i) => Math.min(i + 1, totalFrames - 1));
  }, [totalFrames]);

  const stepBack = useCallback(() => {
    setFrameIndex((i) => Math.max(i - 1, 0));
  }, []);

  // RAF-based playback loop
  useEffect(() => {
    if (!playing) return;

    const interval = frameIntervalMs / speed;

    const tick = (now: number) => {
      if (now - lastTickRef.current >= interval) {
        lastTickRef.current = now;
        setFrameIndex((i) => {
          if (i >= totalFrames - 1) {
            setPlaying(false);
            return i;
          }
          return i + 1;
        });
      }
      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [playing, speed, totalFrames, frameIntervalMs]);

  return {
    frameIndex,
    playing,
    speed,
    setSpeed,
    play,
    stop,
    toggle,
    seek,
    stepForward,
    stepBack,
  };
}

import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Format nanoseconds offset as mm:ss.d */
export function fmtNs(ns: number): string {
  const s = ns / 1e9;
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(1);
  return m > 0 ? `${m}:${sec.padStart(4, "0")}` : `${sec}s`;
}

/** Given a sorted list of timestamps and a target, return the index of the nearest one */
export function nearestIndex(timestamps: number[], target: number): number {
  if (!timestamps.length) return 0;
  let lo = 0, hi = timestamps.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (timestamps[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(timestamps[lo - 1] - target) < Math.abs(timestamps[lo] - target)) {
    return lo - 1;
  }
  return lo;
}

/** Given a sorted list and a target, return the nearest timestamp value */
export function nearestTs(timestamps: number[], target: number): number {
  return timestamps[nearestIndex(timestamps, target)] ?? target;
}

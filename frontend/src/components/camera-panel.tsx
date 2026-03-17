"use client";

import Image from "next/image";
import { api } from "@/lib/api";
import { nearestTs } from "@/lib/utils";
import type { SceneInfo } from "@/lib/types";
import { cn } from "@/lib/utils";

// Ordered camera display — most important first
const CAM_ORDER = [
  "ring_front_center",
  "ring_front_left",
  "ring_front_right",
  "ring_rear_left",
  "ring_rear_right",
  "ring_side_left",
  "ring_side_right",
];

const CAM_LABELS: Record<string, string> = {
  ring_front_center: "Front Center",
  ring_front_left: "Front Left",
  ring_front_right: "Front Right",
  ring_rear_left: "Rear Left",
  ring_rear_right: "Rear Right",
  ring_side_left: "Side Left",
  ring_side_right: "Side Right",
};

interface CameraPanelProps {
  scene: SceneInfo;
  currentTs: number;
  /** If true, show only the 4 most important cameras in compact grid */
  compact?: boolean;
}

export function CameraPanel({ scene, currentTs, compact = false }: CameraPanelProps) {
  const available = CAM_ORDER.filter((c) => scene.camera_timestamps[c]);
  const cameras = compact ? available.slice(0, 4) : available;

  return (
    <div
      className={cn(
        "grid gap-1",
        compact
          ? "grid-cols-2 grid-rows-2"
          : "grid-cols-2 sm:grid-cols-3 lg:grid-cols-4",
      )}
    >
      {cameras.map((cam) => {
        const ts = nearestTs(scene.camera_timestamps[cam] ?? [], currentTs);
        const url = api.cameraUrl(scene.log_id, cam, ts);
        const isCenter = cam === "ring_front_center";

        return (
          <div
            key={cam}
            className={cn(
              "relative bg-black rounded overflow-hidden",
              isCenter && !compact && "col-span-2",
            )}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={url}
              alt={cam}
              className="w-full h-full object-cover cam-img"
              style={{ aspectRatio: "16/9" }}
            />
            <div className="absolute bottom-0 left-0 right-0 px-1.5 py-0.5 bg-black/60 text-[10px] text-white/70">
              {CAM_LABELS[cam] ?? cam}
            </div>
          </div>
        );
      })}
    </div>
  );
}

import { api } from "@/lib/api";
import { PlaybackClient } from "./playback-client";

interface Props {
  params: { logId: string };
}

export const dynamic = "force-dynamic";

export default async function ScenePage({ params }: Props) {
  const scene = await api.scene(params.logId);
  return <PlaybackClient scene={scene} />;
}

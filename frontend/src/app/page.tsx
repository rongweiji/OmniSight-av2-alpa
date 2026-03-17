import Link from "next/link";
import { api } from "@/lib/api";
import { MapPin, Layers, Clock, ChevronRight, AlertCircle } from "lucide-react";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  let scenes: string[] = [];
  let dataDir = "";
  let error = "";

  try {
    const data = await api.scenes();
    scenes = data.scenes;
    dataDir = data.data_dir;
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    error = `API fetch failed — ${msg} (API_URL=${process.env.API_URL ?? "not set, default http://localhost:8080"})`;
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 h-14 flex items-center gap-3">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-md bg-primary/20 border border-primary/30 flex items-center justify-center">
              <Layers className="w-4 h-4 text-primary" />
            </div>
            <span className="font-semibold text-foreground">OmniSight</span>
          </div>
          <span className="text-muted-foreground text-sm">AV2 Dataset Viewer</span>
          <div className="ml-auto text-xs text-muted-foreground font-mono">{dataDir}</div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-10">
        {error ? (
          <div className="flex items-center gap-3 p-4 rounded-lg border border-red-500/30 bg-red-500/10 text-red-400">
            <AlertCircle className="w-5 h-5 shrink-0" />
            <div>
              <p className="font-medium">API not reachable</p>
              <p className="text-sm mt-0.5">{error}</p>
              <p className="text-sm mt-1 text-red-300/70">
                Start the backend: <code className="bg-red-500/10 px-1 rounded">python -m api.server --data-dir /raid/av2/sensor/val</code>
              </p>
            </div>
          </div>
        ) : (
          <>
            <div className="mb-8">
              <h1 className="text-2xl font-bold text-foreground">Scenes</h1>
              <p className="text-muted-foreground mt-1">
                {scenes.length} scene{scenes.length !== 1 ? "s" : ""} available — click to open the playback viewer
              </p>
            </div>

            {scenes.length === 0 ? (
              <div className="text-center py-20 text-muted-foreground">
                <Layers className="w-12 h-12 mx-auto mb-4 opacity-30" />
                <p className="font-medium">No scenes found</p>
                <p className="text-sm mt-1">Download AV2 data into <code className="text-primary">{dataDir}</code></p>
              </div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {scenes.map((id) => (
                  <SceneCard key={id} logId={id} />
                ))}
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}

async function SceneCard({ logId }: { logId: string }) {
  let info = null;
  try {
    info = await api.scene(logId);
  } catch {
    // show minimal card if metadata fetch fails
  }

  return (
    <Link
      href={`/scene/${logId}`}
      className="group block rounded-xl border border-border bg-card hover:border-primary/50 hover:bg-card/80 transition-all duration-200 overflow-hidden"
    >
      <div className="p-4">
        <div className="flex items-start justify-between gap-2 mb-3">
          <code className="text-xs text-primary font-mono break-all leading-relaxed">
            {logId.slice(0, 36)}
          </code>
          <ChevronRight className="w-4 h-4 text-muted-foreground group-hover:text-primary shrink-0 mt-0.5 transition-colors" />
        </div>

        {info ? (
          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <MapPin className="w-3 h-3" />
              <span>{info.city_name}</span>
            </div>
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Clock className="w-3 h-3" />
              <span>{info.duration_s}s &nbsp;·&nbsp; {info.n_lidar_frames} frames</span>
            </div>
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Layers className="w-3 h-3" />
              <span>{info.n_annotations} annotations &nbsp;·&nbsp; {info.camera_names.length} cameras</span>
            </div>
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">Loading metadata…</p>
        )}
      </div>

      <div className="px-4 py-2.5 border-t border-border bg-accent/30 text-xs text-primary font-medium group-hover:bg-primary/10 transition-colors">
        Open playback viewer →
      </div>
    </Link>
  );
}

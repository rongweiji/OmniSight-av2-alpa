export interface SceneListResponse {
  scenes: string[];
  count: number;
  data_dir: string;
}

export interface SceneInfo {
  log_id: string;
  city_name: string;
  lidar_timestamps: number[];
  camera_timestamps: Record<string, number[]>;
  camera_names: string[];
  n_lidar_frames: number;
  n_annotations: number;
  duration_s: number;
}

export interface LidarBounds {
  x: [number, number];
  y: [number, number];
  z: [number, number];
}

export interface LidarFrame {
  timestamp_ns: number;
  n_points: number;
  points: [number, number, number][];
  intensity: number[] | null;
  bounds: LidarBounds;
}

export interface Annotation {
  category: string;
  color: string;
  track_uuid: string;
  x: number;
  y: number;
  z: number;
  length: number;
  width: number;
  height: number;
}

export interface AnnotationsResponse {
  timestamp_ns: number;
  annotations: Annotation[];
}

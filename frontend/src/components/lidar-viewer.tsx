"use client";

import { useMemo, useRef } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera, Grid } from "@react-three/drei";
import * as THREE from "three";
import type { LidarFrame, Annotation } from "@/lib/types";

// ── Point cloud ───────────────────────────────────────────────────────────────

function PointCloud({ frame }: { frame: LidarFrame }) {
  const { positions, colors } = useMemo(() => {
    const pts = frame.points;
    const n = pts.length;
    const positions = new Float32Array(n * 3);
    const colors = new Float32Array(n * 3);

    const zMin = frame.bounds.z[0];
    const zMax = frame.bounds.z[1];
    const zRange = (zMax - zMin) || 1;

    for (let i = 0; i < n; i++) {
      const [ax, ay, az] = pts[i];
      // AV2: x=forward, y=left, z=up → Three.js: x=right, y=up, z=back
      positions[i * 3 + 0] = ax;
      positions[i * 3 + 1] = az;
      positions[i * 3 + 2] = -ay;

      // Colour by height: blue (low) → cyan → green → yellow (high)
      const t = (az - zMin) / zRange;
      const c = new THREE.Color();
      c.setHSL(0.65 - t * 0.55, 1.0, 0.45 + t * 0.1);
      colors[i * 3 + 0] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }

    return { positions, colors };
  }, [frame]);

  return (
    <points>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial size={0.12} vertexColors sizeAttenuation />
    </points>
  );
}

// ── Annotation boxes ──────────────────────────────────────────────────────────

function AnnotationBoxes({ annotations }: { annotations: Annotation[] }) {
  return (
    <>
      {annotations.map((ann, i) => (
        <mesh
          key={`${ann.track_uuid}-${i}`}
          position={[ann.x, ann.z + ann.height / 2, -ann.y]}
        >
          <boxGeometry args={[ann.length, ann.height, ann.width]} />
          <meshBasicMaterial color={ann.color} wireframe transparent opacity={0.85} />
        </mesh>
      ))}
    </>
  );
}

// ── Ego vehicle marker ────────────────────────────────────────────────────────

function EgoMarker() {
  return (
    <group>
      {/* Car body */}
      <mesh position={[0, 0.75, 0]}>
        <boxGeometry args={[2, 1.5, 4.5]} />
        <meshBasicMaterial color="#ef4444" wireframe />
      </mesh>
      {/* Direction arrow */}
      <arrowHelper
        args={[
          new THREE.Vector3(0, 0, -1),
          new THREE.Vector3(0, 0.75, -2.5),
          3,
          0xef4444,
        ]}
      />
    </group>
  );
}

// ── Camera reset on new frame ─────────────────────────────────────────────────

function CameraSetup() {
  const { camera } = useThree();
  useMemo(() => {
    camera.position.set(0, 25, 0.1);
    camera.lookAt(0, 0, 0);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  return null;
}

// ── Main export ───────────────────────────────────────────────────────────────

interface LidarViewerProps {
  frame: LidarFrame | null;
  annotations?: Annotation[];
}

export function LidarViewer({ frame, annotations = [] }: LidarViewerProps) {
  if (!frame) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-black/60 rounded-lg text-muted-foreground text-sm">
        Loading LiDAR…
      </div>
    );
  }

  return (
    <Canvas className="rounded-lg" gl={{ antialias: false }}>
      <CameraSetup />
      <PerspectiveCamera makeDefault fov={60} near={0.1} far={500} />
      <OrbitControls
        enablePan
        enableZoom
        enableRotate
        minDistance={5}
        maxDistance={200}
      />

      <ambientLight intensity={0.3} />
      <Grid
        args={[200, 200]}
        cellColor="#1e293b"
        sectionColor="#334155"
        sectionSize={10}
        fadeDistance={150}
      />

      <PointCloud frame={frame} />
      <AnnotationBoxes annotations={annotations} />
      <EgoMarker />
    </Canvas>
  );
}

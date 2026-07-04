"use client";

import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Bounds, PerspectiveCamera } from "@react-three/drei";
import * as THREE from "three";
import { useCursorTrack } from "@/hooks/useCursorTrack";

const MAX_TILT_RAD = THREE.MathUtils.degToRad(15);
const PULSE_SPEED = 0.05; // fraction of the loop travelled per second

// Each pulse takes a real 2026 team colour — the pit-wall watches the
// whole grid, not just one car.
const PULSE_COLORS = ["#e80020", "#ff8000", "#27f4d2", "#0090ff", "#3671c6"];

// Hand-authored closed-loop control points: start/finish straight, a fast
// kink, a flowing esses section, a hairpin, a long back straight, and a
// sweeping final corner. Not a real track — a generative "circuit DNA"
// so the hero needs no licensed geodata or 3D assets.
const CIRCUIT_POINTS: [number, number, number][] = [
  [0, 0, -3],
  [2.2, 0, -3],
  [3.2, 0.1, -1.8],
  [2.6, 0.15, -0.6],
  [3.4, 0.05, 0.6],
  [2.4, 0.1, 1.4],
  [0.8, 0, 2.2],
  [-0.6, 0, 2.6],
  [-1.8, 0, 2.0],
  [-3.0, 0.05, 0.9],
  [-3.4, 0.1, -0.8],
  [-2.6, 0, -2.0],
  [-1.2, 0, -2.8],
];

function PulseDots({ curve }: { curve: THREE.CatmullRomCurve3 }) {
  const dotsRef = useRef<(THREE.Group | null)[]>([]);
  const phases = useMemo(
    () => PULSE_COLORS.map((_, i) => i / PULSE_COLORS.length),
    []
  );
  const tRef = useRef(0);

  useFrame((_, delta) => {
    tRef.current = (tRef.current + delta * PULSE_SPEED) % 1;
    dotsRef.current.forEach((dot, i) => {
      if (!dot) return;
      const t = (tRef.current + phases[i]) % 1;
      dot.position.copy(curve.getPointAt(t));
    });
  });

  return (
    <>
      {PULSE_COLORS.map((color, i) => (
        <group
          key={color}
          ref={(el) => {
            dotsRef.current[i] = el;
          }}
        >
          <mesh>
            <sphereGeometry args={[0.045, 12, 12]} />
            <meshBasicMaterial color={color} toneMapped={false} />
          </mesh>
          <mesh>
            <sphereGeometry args={[0.12, 12, 12]} />
            <meshBasicMaterial
              color={color}
              transparent
              opacity={0.35}
              blending={THREE.AdditiveBlending}
              depthWrite={false}
            />
          </mesh>
        </group>
      ))}
    </>
  );
}

function Circuit() {
  const group = useRef<THREE.Group>(null);
  const { tick } = useCursorTrack();

  const { curve, tubeGeometry } = useMemo(() => {
    const c = new THREE.CatmullRomCurve3(
      CIRCUIT_POINTS.map(([x, y, z]) => new THREE.Vector3(x, y, z)),
      true,
      "catmullrom",
      0.5
    );
    const tube = new THREE.TubeGeometry(c, 300, 0.03, 8, true);
    return { curve: c, tubeGeometry: tube };
  }, []);

  useFrame((state) => {
    const { x, y } = tick(0.05);
    if (group.current) {
      group.current.rotation.y = x * MAX_TILT_RAD;
      group.current.rotation.x = y * MAX_TILT_RAD;
      group.current.position.y = Math.sin(state.clock.elapsedTime * 0.4) * 0.06;
    }
  });

  return (
    <group ref={group}>
      <mesh geometry={tubeGeometry}>
        <meshBasicMaterial color="#ffffff" wireframe transparent opacity={0.16} />
      </mesh>
      <PulseDots curve={curve} />
    </group>
  );
}

export default function CircuitCanvas() {
  return (
    <Canvas
      dpr={[1, 1.5]}
      gl={{ antialias: true, alpha: true }}
      className="!absolute inset-0"
    >
      <PerspectiveCamera makeDefault position={[0, 6.4, 8.2]} fov={42} />
      <Bounds fit clip observe margin={1.3}>
        <Circuit />
      </Bounds>
    </Canvas>
  );
}

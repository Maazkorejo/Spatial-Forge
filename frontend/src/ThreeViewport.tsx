import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

interface ThreeViewportProps {
  mesh?: {
    vertices: [number, number, number][];
    faces: [number, number, number][];
  };
  viewMode: 'shaded' | 'wireframe' | 'solid';
  cutsCount?: number;
}

export const ThreeViewport: React.FC<ThreeViewportProps> = ({ mesh, viewMode }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const meshGroupRef = useRef<THREE.Group | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Initialize Scene, Camera, Renderer, Lighting
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const width = container.clientWidth || 600;
    const height = container.clientHeight || 450;

    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 2000);
    camera.position.set(160, 140, 200);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    rendererRef.current = renderer;

    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.maxDistance = 1200;
    controls.minDistance = 20;
    controlsRef.current = controls;

    // Lighting rig
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x1e2738, 0.5);
    scene.add(hemiLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.9);
    dirLight1.position.set(120, 180, 140);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x7ea2cc, 0.45);
    dirLight2.position.set(-120, -60, -100);
    scene.add(dirLight2);

    // Subtle CAD Ground Grid
    const grid = new THREE.GridHelper(300, 30, 0x7a93b4, 0xb8cbdf);
    grid.position.y = -0.5;
    (grid.material as THREE.Material).opacity = 0.35;
    (grid.material as THREE.Material).transparent = true;
    scene.add(grid);

    const meshGroup = new THREE.Group();
    scene.add(meshGroup);
    meshGroupRef.current = meshGroup;

    // Render loop
    const animate = () => {
      animationFrameRef.current = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Resize observer
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = entry.contentRect.width;
        const h = entry.contentRect.height;
        if (w > 0 && h > 0) {
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
          renderer.setSize(w, h);
        }
      }
    });
    resizeObserver.observe(container);

    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      resizeObserver.disconnect();
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement.parentElement) {
        renderer.domElement.parentElement.removeChild(renderer.domElement);
      }
    };
  }, []);

  // Update Geometry and Materials when mesh or viewMode changes
  useEffect(() => {
    const meshGroup = meshGroupRef.current;
    if (!meshGroup || !mesh || !mesh.vertices || mesh.vertices.length === 0) return;

    // Clear previous children
    while (meshGroup.children.length > 0) {
      const child = meshGroup.children[0] as THREE.Mesh;
      if (child.geometry) child.geometry.dispose();
      meshGroup.remove(child);
    }

    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(mesh.vertices.flat());
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const indices = new Uint32Array(mesh.faces.flat());
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    geometry.computeVertexNormals();

    // Center geometry on origin
    geometry.computeBoundingBox();
    const bbox = geometry.boundingBox!;
    const center = new THREE.Vector3();
    bbox.getCenter(center);
    geometry.translate(-center.x, -bbox.min.y, -center.z);

    let material: THREE.Material;
    if (viewMode === 'wireframe') {
      material = new THREE.MeshBasicMaterial({
        color: 0x3d65aa,
        wireframe: true,
      });
    } else if (viewMode === 'solid') {
      material = new THREE.MeshLambertMaterial({
        color: 0x5a7ca8,
      });
    } else {
      // 'shaded' PBR
      material = new THREE.MeshStandardMaterial({
        color: 0x557b9f,
        roughness: 0.35,
        metalness: 0.3,
        polygonOffset: true,
        polygonOffsetFactor: 1,
        polygonOffsetUnits: 1,
      });
    }

    const threeMesh = new THREE.Mesh(geometry, material);
    meshGroup.add(threeMesh);

    // Add crisp edge contours if not pure wireframe
    if (viewMode !== 'wireframe') {
      const edges = new THREE.EdgesGeometry(geometry, 24);
      const edgeLine = new THREE.LineSegments(
        edges,
        new THREE.LineBasicMaterial({ color: 0x1d3048, linewidth: 1, transparent: true, opacity: 0.55 })
      );
      meshGroup.add(edgeLine);
    }

    // Adjust camera target to center
    if (controlsRef.current && cameraRef.current) {
      const size = new THREE.Vector3();
      bbox.getSize(size);
      const maxDim = Math.max(size.x, size.y, size.z, 20);
      controlsRef.current.target.set(0, size.y / 2, 0);
      cameraRef.current.position.set(maxDim * 1.5, maxDim * 1.2, maxDim * 1.7);
      controlsRef.current.update();
    }
  }, [mesh, viewMode]);

  return <div ref={containerRef} style={{ width: '100%', height: '100%', position: 'absolute', inset: 0 }} />;
};

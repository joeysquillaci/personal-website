import * as THREE from "https://esm.sh/three@0.160.0";
import { STLLoader } from "https://esm.sh/three@0.160.0/examples/jsm/loaders/STLLoader.js";

const stlViewers = document.querySelectorAll(".stl-viewer");

if (window.location.protocol === "file:") {
  stlViewers.forEach((viewer) => {
    const status = viewer.querySelector(".stl-viewer__status");
    if (status) {
      status.textContent =
        "STL previews require a local web server. Open this site from http://localhost instead of file://.";
    }
  });
} else {
stlViewers.forEach((viewer) => {
  const modelSrc = viewer.dataset.stlSrc;
  const modelName = viewer.dataset.modelName || "3D model";
  const status = viewer.querySelector(".stl-viewer__status");

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x07111f);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
  camera.position.set(0, 0, 140);

  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
  });

  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  viewer.appendChild(renderer.domElement);

  const ambientLight = new THREE.AmbientLight(0xffffff, 1.8);
  scene.add(ambientLight);

  const directionalLight = new THREE.DirectionalLight(0x77a9ff, 2.5);
  directionalLight.position.set(40, 60, 80);
  scene.add(directionalLight);

  const fillLight = new THREE.DirectionalLight(0x45e0c6, 1.4);
  fillLight.position.set(-30, -10, 60);
  scene.add(fillLight);

  const material = new THREE.MeshStandardMaterial({
    color: 0xc8d7ff,
    metalness: 0.18,
    roughness: 0.42,
  });

  const modelGroup = new THREE.Group();
  scene.add(modelGroup);

  let isDragging = false;
  let previousPointer = { x: 0, y: 0 };
  let targetRotation = { x: -0.55, y: 0.75 };
  let currentRotation = { x: -0.55, y: 0.75 };
  let distance = 140;

  function setStatus(message) {
    if (status) {
      status.textContent = message;
    }
  }

  function updateSize() {
    const width = viewer.clientWidth;
    const height = viewer.clientHeight;

    if (!width || !height) {
      return;
    }

    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }

  function fitCameraToMesh(targetMesh) {
    const box = new THREE.Box3().setFromObject(targetMesh);
    const size = box.getSize(new THREE.Vector3());
    const maxDimension = Math.max(size.x, size.y, size.z) || 1;

    targetMesh.geometry.center();
    distance = Math.max(maxDimension * 1.8, 55);
    camera.position.set(0, 0, distance);
    camera.lookAt(0, 0, 0);
  }

  function animate() {
    currentRotation.x += (targetRotation.x - currentRotation.x) * 0.08;
    currentRotation.y += (targetRotation.y - currentRotation.y) * 0.08;

    modelGroup.rotation.x = currentRotation.x;
    modelGroup.rotation.y = currentRotation.y;

    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }

  const loader = new STLLoader();

  loader.load(
    encodeURI(modelSrc),
    (geometry) => {
      geometry.computeVertexNormals();
      const mesh = new THREE.Mesh(geometry, material);
      modelGroup.add(mesh);
      fitCameraToMesh(mesh);
      viewer.classList.add("is-ready");
      setStatus(`${modelName} loaded. Drag to inspect the model.`);
    },
    undefined,
    () => {
      setStatus(`Could not load ${modelSrc}.`);
    }
  );

  viewer.addEventListener("pointerdown", (event) => {
    isDragging = true;
    previousPointer = {
      x: event.clientX,
      y: event.clientY,
    };
    viewer.setPointerCapture(event.pointerId);
  });

  viewer.addEventListener("pointermove", (event) => {
    if (!isDragging) {
      return;
    }

    const deltaX = event.clientX - previousPointer.x;
    const deltaY = event.clientY - previousPointer.y;

    targetRotation.y += deltaX * 0.01;
    targetRotation.x += deltaY * 0.01;
    targetRotation.x = Math.max(-1.4, Math.min(1.4, targetRotation.x));

    previousPointer = {
      x: event.clientX,
      y: event.clientY,
    };
  });

  viewer.addEventListener("pointerup", (event) => {
    isDragging = false;
    viewer.releasePointerCapture(event.pointerId);
  });

  viewer.addEventListener("pointerleave", () => {
    isDragging = false;
  });

  viewer.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      distance += event.deltaY * 0.08;
      distance = Math.max(35, Math.min(260, distance));
      camera.position.z = distance;
    },
    { passive: false }
  );

  const resizeObserver = new ResizeObserver(() => {
    updateSize();
  });

  resizeObserver.observe(viewer);
  updateSize();
  animate();
});
}

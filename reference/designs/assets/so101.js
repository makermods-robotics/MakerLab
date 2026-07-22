/* Reusable SO-101 URDF viewer. ES module.
   usage: import { mountSO101 } from './assets/so101.js';
          mountSO101(canvas, { mode: 'idle' | 'teleop' });
   Page must include the importmap for three/urdf-loader (see collect.html). */

export async function mountSO101(canvas, opts = {}) {
  const THREE = await import('three');
  const { default: URDFLoader } = await import('urdf-loader');

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 0.01, 20);
  camera.position.set(...(opts.camera || [0.5, 0.3, 0.62]));
  camera.lookAt(0, 0.15, 0);

  const dark = !matchMedia('(prefers-color-scheme: light)').matches;
  const armColor = new THREE.Color(dark ? 0xc8c8c8 : 0x4a4a4a);

  scene.add(new THREE.AmbientLight(0xffffff, dark ? 0.55 : 0.9));
  const key = new THREE.DirectionalLight(0xffffff, dark ? 1.6 : 2.2);
  key.position.set(1.2, 1.8, 1.4);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xff7a40, dark ? 2.4 : 1.2);
  rim.position.set(-1.6, 0.7, -1.2);
  scene.add(rim);

  const grid = new THREE.GridHelper(3, 42, 0xff7a40, dark ? 0x2a2a2a : 0xd9d9d9);
  grid.material.transparent = true;
  grid.material.opacity = dark ? 0.5 : 0.7;
  scene.add(grid);

  const loader = new URDFLoader();
  loader.packages = { so_arm_description: '/frontend/public/so-101-urdf' };
  const robot = await new Promise((res, rej) => {
    loader.load('/frontend/public/so-101-urdf/urdf/so101_new_calib.urdf', res, undefined, rej);
  });
  robot.rotation.x = -Math.PI / 2;
  scene.add(robot);

  const armMat = new THREE.MeshStandardMaterial({ color: armColor, metalness: 0.25, roughness: 0.55 });
  const skin = () => robot.traverse((o) => { if (o.isMesh && o.material !== armMat) o.material = armMat; });

  const J = (n, v) => robot.joints[n] && robot.joints[n].setJointValue(v);
  const poses = {
    idle: (t) => {
      J('Rotation',    0.38 * Math.sin(t * 0.40));
      J('Pitch',      -0.28 + 0.08 * Math.sin(t * 0.60 + 1.2));
      J('Elbow',       0.62 + 0.10 * Math.sin(t * 0.52 + 0.4));
      J('Wrist_Pitch', 0.92 + 0.14 * Math.sin(t * 0.70 + 2.1));
      J('Wrist_Roll',  0.30 * Math.sin(t * 0.33));
      J('Jaw',         0.30 + 0.26 * Math.sin(t * 0.90 + 3.0));
    },
    // livelier — reads as "being driven"
    teleop: (t) => {
      J('Rotation',    0.65 * Math.sin(t * 1.1));
      J('Pitch',      -0.35 + 0.22 * Math.sin(t * 1.4 + 1.2));
      J('Elbow',       0.70 + 0.25 * Math.sin(t * 1.25 + 0.4));
      J('Wrist_Pitch', 0.80 + 0.28 * Math.sin(t * 1.6 + 2.1));
      J('Wrist_Roll',  0.55 * Math.sin(t * 0.9));
      J('Jaw',         0.35 + 0.32 * Math.sin(t * 2.2 + 3.0));
    },
  };
  let mode = opts.mode || 'idle';

  const host = canvas.parentElement;
  const resize = () => {
    const w = host.clientWidth, h = host.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  addEventListener('resize', resize);
  resize();

  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reduced) {
    poses[mode](2.1);
    const draw = () => { skin(); renderer.render(scene, camera); };
    setTimeout(draw, 300); setTimeout(draw, 1500); draw();
  } else {
    renderer.setAnimationLoop((ms) => {
      skin();
      poses[mode](ms / 1000);
      renderer.render(scene, camera);
    });
  }

  return { setMode: (m) => { if (poses[m]) mode = m; } };
}

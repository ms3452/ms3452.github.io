/**
 * 3D overview of the programmable waveguide setup (Three.js): a projector writes the
 * refractive index pattern through a lens onto the chip, a high-voltage amplifier drives its
 * top and bottom electrodes, and fibers carry the input and output light. The chip's top face
 * shows the live 2D simulation canvas as a texture.
 *
 * Fibers follow the demo's ports (state.demo.ports: [center, width] in um). Their brightness
 * is proportional to the power they carry: an input fiber the share of the input launched
 * through it, an output fiber the power coupled into its mode (the detector readout).
 *
 * Schematic proportions: the chip is stretched across x like the 2D view, and the
 * projector and lens are not to scale.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

// Scene units: the chip is CHIP_L long along propagation z (3D +x axis) and CHIP_W wide
// along x (3D z axis; x = -Lx/2 lies at the far side, matching the top of the 2D view)
const CHIP_L = 8.0;
const CHIP_W = 3.0;
const CHIP_T = 0.35; // substrate thickness
const CHIP_TOP = 0.0; // y of the waveguide layer
const LENS_Y = 4.2;
const PROJECTOR_Y = 7.6;
// Colors of the 3D scene per 2D palette (palette select): the guided light (fibers, light
// sources, bar glow), the projector light that writes the index (green for the palettes that
// show the index in green), the MNIST bars and the winning bar
const BLUE_PROJECTOR = '#9fb4ff';
const GREEN_PROJECTOR = '#4ade80';
const PALETTES = {
  cyan_gold: { light: '#00f2fe', projector: BLUE_PROJECTOR, bar: '#0891b2', winner: '#d97706', winnerGlow: '#f59e0b' },
  inferno: { light: '#fb8b24', projector: BLUE_PROJECTOR, bar: '#9a3412', winner: '#fde047', winnerGlow: '#fef08a' },
  viridis: { light: '#35b779', projector: BLUE_PROJECTOR, bar: '#21918c', winner: '#fde725', winnerGlow: '#fde725' },
  green_laser: { light: '#39ff6a', projector: BLUE_PROJECTOR, bar: '#15803d', winner: '#d9f99d', winnerGlow: '#bef264' },
  green_index_red_light: { light: '#ff2a2a', projector: GREEN_PROJECTOR, bar: '#991b1b', winner: '#fde047', winnerGlow: '#facc15' },
  green_index_thermal_light: { light: '#ff7a1a', projector: GREEN_PROJECTOR, bar: '#9d174d', winner: '#fde047', winnerGlow: '#fef08a' },
};
const FIBER_GAIN = 2.6; // emissive intensity for unit power (all of one input's light)
const FIBER_FLOOR = 0.03; // dark fibers stay faintly visible
const TABLE_TOP = -1.75;
const BAR_HEIGHT = 10.0; // MNIST detector bars: height for all of the detected light in one bin
// The camera orbits the center of the chip's top face (the chip is centered on the origin)
const TARGET_HOME = new THREE.Vector3(0, CHIP_TOP, 0);
// Camera 15 deg above the table (75 deg from vertical); only the azimuth and distance change.
// The distance keeps the projector above the chip in view.
const CAMERA_POLAR = THREE.MathUtils.degToRad(75);
const CAMERA_HOME = new THREE.Vector3()
  .setFromSphericalCoords(31.0, CAMERA_POLAR, Math.atan2(-7.5, 19.0))
  .add(TARGET_HOME);

export class Scene3D {
  constructor(container, sourceCanvas) {
    this.container = container;
    this.palette = {};
    this.projectorMaterials = []; // aperture + projected beam, recolored with the palette
    this._setPaletteColors('cyan_gold');

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    container.appendChild(this.renderer.domElement);

    this.labelRenderer = new CSS2DRenderer();
    this.labelRenderer.domElement.className = 'view3d-labels';
    container.appendChild(this.labelRenderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#080c14');
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.6;

    this.camera = new THREE.PerspectiveCamera(40, 1, 0.1, 200);
    this.camera.position.copy(CAMERA_HOME);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.copy(TARGET_HOME);
    this.controls.enableDamping = true;
    this.controls.minDistance = 3;
    this.controls.maxDistance = 40;
    // Only zoom and azimuthal rotation: no panning, polar angle fixed
    this.controls.enablePan = false;
    this.controls.minPolarAngle = CAMERA_POLAR;
    this.controls.maxPolarAngle = CAMERA_POLAR;

    const keyLight = new THREE.DirectionalLight('#ffffff', 1.6);
    keyLight.position.set(-6, 10, 8);
    this.scene.add(keyLight, new THREE.AmbientLight('#8090b0', 0.4));

    // Live simulation canvas -> chip texture (re-uploaded every frame right after the 2D draw)
    this.chipTexture = new THREE.CanvasTexture(sourceCanvas);
    this.chipTexture.colorSpace = THREE.SRGBColorSpace;
    this.chipTexture.anisotropy = 8;

    this._buildBench();
    this._buildChip();
    this._buildProjector();
    this._buildLens();
    this._buildBeam();
    this._buildElectrodes();

    this.pulseTexture = this._makePulseTexture();
    this.fiberGroup = new THREE.Group();
    this.scene.add(this.fiberGroup);
    this.inputFibers = [];
    this.outputFibers = [];
    this.inputShares = []; // [input][fiber]: share of each input's power in each input fiber
    this.sources = []; // MNIST: one integrated light source per input beam
    this.bars = []; // MNIST: one bar per digit bin at the end of its output fiber
    this.digitSprite = null; // MNIST: the input image next to the sources
    this.fiberKey = null;
    this.pulsedShown = null;

    this._resize();
    if (window.ResizeObserver) new ResizeObserver(() => this._resize()).observe(container);
  }

  /** Recolor the light, projector and MNIST bars to match the 2D palette `name`. */
  setPalette(name) {
    this._setPaletteColors(name);
    for (const m of this.projectorMaterials) m.color.copy(this.palette.projector);
    for (const f of [...this.inputFibers, ...this.outputFibers]) f.core.material.emissive.copy(this.palette.light);
    // Light sources and bars are recolored every frame in _updatePorts
  }

  _setPaletteColors(name) {
    const p = PALETTES[name] || PALETTES.cyan_gold;
    for (const k of Object.keys(p)) this.palette[k] = new THREE.Color(p[k]);
  }

  _projectorMaterial(material) {
    this.projectorMaterials.push(material);
    return material;
  }

  resetView() {
    this.camera.position.copy(CAMERA_HOME);
    this.controls.target.copy(TARGET_HOME);
  }

  // ------------------------------------------------------------------ scene parts
  _label(text, x, y, z, parent = this.scene) {
    const div = document.createElement('div');
    div.className = 'label3d';
    div.textContent = text;
    const label = new CSS2DObject(div);
    label.position.set(x, y, z);
    parent.add(label);
    return label;
  }

  _buildBench() {
    // Optical table: dark plate with a faint grid of tapped holes
    const table = new THREE.Mesh(
      new THREE.BoxGeometry(34, 0.3, 18),
      new THREE.MeshStandardMaterial({ color: '#1a2130', roughness: 0.6, metalness: 0.5 }),
    );
    table.position.y = -1.9;
    this.scene.add(table);
    const grid = new THREE.GridHelper(34, 34, '#2a3448', '#222b3d');
    grid.position.y = -1.74;
    this.scene.add(grid);

    // Chip mount (pedestal)
    const mount = new THREE.Mesh(
      new THREE.BoxGeometry(CHIP_L * 0.8, 1.4, CHIP_W * 0.9),
      new THREE.MeshStandardMaterial({ color: '#6b7280', roughness: 0.35, metalness: 0.9 }),
    );
    mount.position.y = CHIP_TOP - CHIP_T - 0.03 - 0.7; // below the bottom electrode
    this.scene.add(mount);
  }

  _buildChip() {
    const substrate = new THREE.Mesh(
      new THREE.BoxGeometry(CHIP_L, CHIP_T, CHIP_W + 0.5),
      new THREE.MeshStandardMaterial({ color: '#2b2f3a', roughness: 0.25, metalness: 0.7 }),
    );
    substrate.position.y = CHIP_TOP - CHIP_T / 2 - 0.005;
    this.scene.add(substrate);

    // Waveguide layer: unlit so the colors match the 2D view exactly
    const layer = new THREE.Mesh(
      new THREE.PlaneGeometry(CHIP_L, CHIP_W),
      new THREE.MeshBasicMaterial({ map: this.chipTexture, toneMapped: false }),
    );
    layer.rotation.x = -Math.PI / 2;
    layer.position.y = CHIP_TOP;
    this.scene.add(layer);

    this._label('Programmable waveguide chip', 0, CHIP_TOP - 0.3, CHIP_W / 2 + 0.9);
  }

  _buildProjector() {
    const metal = new THREE.MeshStandardMaterial({ color: '#d1d5db', roughness: 0.3, metalness: 0.8 });
    const dark = new THREE.MeshStandardMaterial({ color: '#111827', roughness: 0.5, metalness: 0.4 });

    const body = new THREE.Mesh(new THREE.BoxGeometry(3.4, 1.4, 2.4), metal);
    body.position.set(0, PROJECTOR_Y + 0.7, 0);
    const vents = new THREE.Mesh(new THREE.BoxGeometry(3.42, 0.5, 1.6), dark);
    vents.position.set(0, PROJECTOR_Y + 0.85, 0);
    const barrel = new THREE.Mesh(new THREE.CylinderGeometry(0.45, 0.5, 0.8, 40), dark);
    barrel.position.set(0, PROJECTOR_Y - 0.4, 0);
    const aperture = new THREE.Mesh(
      new THREE.CircleGeometry(0.36, 40),
      this._projectorMaterial(new THREE.MeshBasicMaterial({ color: this.palette.projector, toneMapped: false })),
    );
    aperture.rotation.x = Math.PI / 2;
    aperture.position.set(0, PROJECTOR_Y - 0.81, 0);
    this.scene.add(body, vents, barrel, aperture);

    this._label('Projector (writes Δn pattern)', 2.6, PROJECTOR_Y + 1.6, 0);
  }

  _buildLens() {
    // Biconvex lens: lathe of a two-arc profile around the vertical axis
    const radius = 1.7;
    const sag = 0.28;
    const edge = 0.05;
    const points = [];
    const n = 24;
    for (let i = 0; i <= n; i++) {
      const r = (radius * i) / n;
      points.push(new THREE.Vector2(r, -(edge + sag * (1 - (r / radius) ** 2))));
    }
    for (let i = n; i >= 0; i--) {
      const r = (radius * i) / n;
      points.push(new THREE.Vector2(r, edge + sag * (1 - (r / radius) ** 2)));
    }
    const lens = new THREE.Mesh(
      new THREE.LatheGeometry(points, 64),
      new THREE.MeshPhysicalMaterial({
        color: '#e0f2fe', transmission: 1, roughness: 0.04, thickness: 0.6, ior: 1.5,
        transparent: true, opacity: 0.9,
      }),
    );
    lens.position.y = LENS_Y;

    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(radius + 0.05, 0.08, 16, 64),
      new THREE.MeshStandardMaterial({ color: '#111827', roughness: 0.5, metalness: 0.6 }),
    );
    ring.rotation.x = Math.PI / 2;
    ring.position.y = LENS_Y;
    this.scene.add(lens, ring);

    this._label('Lens', -radius - 1.0, LENS_Y, 0);
  }

  _buildBeam() {
    // Projected light: aperture -> lens (diverging), lens -> chip footprint
    const fade = document.createElement('canvas');
    fade.width = 4;
    fade.height = 128;
    const ctx = fade.getContext('2d');
    const g = ctx.createLinearGradient(0, 0, 0, 128);
    g.addColorStop(0, 'rgba(255,255,255,0.10)');
    g.addColorStop(1, 'rgba(255,255,255,0.40)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 4, 128);
    const alphaMap = new THREE.CanvasTexture(fade);

    const beamMaterial = this._projectorMaterial(new THREE.MeshBasicMaterial({
      color: this.palette.projector, transparent: true, opacity: 0.22, alphaMap,
      blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide, toneMapped: false,
    }));
    // Square frustum: 4-sided open cylinder, turned so its faces are axis-aligned
    const frustum = (yTop, yBottom, topHalf, bottomHalfX, bottomHalfZ) => {
      const s = Math.SQRT2;
      const geometry = new THREE.CylinderGeometry(topHalf * s, 1 * s, yTop - yBottom, 4, 1, true);
      geometry.rotateY(Math.PI / 4);
      // Scale only the bottom ring to the (rectangular) footprint
      const pos = geometry.attributes.position;
      for (let i = 0; i < pos.count; i++) {
        if (pos.getY(i) < 0) {
          pos.setX(i, pos.getX(i) * bottomHalfX);
          pos.setZ(i, pos.getZ(i) * bottomHalfZ);
        }
      }
      const mesh = new THREE.Mesh(geometry, beamMaterial);
      mesh.position.y = (yTop + yBottom) / 2;
      return mesh;
    };
    this.scene.add(
      frustum(PROJECTOR_Y - 0.82, LENS_Y + 0.3, 0.3, 1.1, 1.1),
      frustum(LENS_Y - 0.3, CHIP_TOP + 0.01, 1.1, CHIP_L / 2, CHIP_W / 2),
    );
  }

  _buildElectrodes() {
    // One contact to the top electrode and one to the bottom electrode, in the far corner at
    // the input facet, wired to a high-voltage amplifier on the table
    const gold = new THREE.MeshStandardMaterial({ color: '#d4a537', roughness: 0.25, metalness: 1.0 });
    const cornerX = -CHIP_L / 2 + 0.5;
    const edgeZ = -(CHIP_W / 2 + 0.25); // far edge of the substrate
    const topPad = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.03, 0.2), gold);
    topPad.position.set(cornerX, CHIP_TOP + 0.01, edgeZ + 0.13);
    const bottomElectrode = new THREE.Mesh(new THREE.BoxGeometry(CHIP_L, 0.025, CHIP_W + 0.5), gold);
    bottomElectrode.position.y = CHIP_TOP - CHIP_T - 0.018;
    const bottomTab = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.025, 0.5), gold);
    bottomTab.position.set(cornerX, CHIP_TOP - CHIP_T - 0.018, edgeZ - 0.25);
    this.scene.add(topPad, bottomElectrode, bottomTab);

    // Amplifier: box with a display, a knob and two output terminals facing the viewer
    const ampX = -9.0;
    const ampZ = -6.0;
    const amp = new THREE.Mesh(
      new THREE.BoxGeometry(3.2, 1.5, 2.2),
      new THREE.MeshStandardMaterial({ color: '#374151', roughness: 0.45, metalness: 0.6 }),
    );
    amp.position.set(ampX, TABLE_TOP + 0.75, ampZ);
    const front = ampZ + 1.1;
    const cv = document.createElement('canvas');
    cv.width = 256;
    cv.height = 96;
    const ctx = cv.getContext('2d');
    ctx.fillStyle = '#05140f';
    ctx.fillRect(0, 0, 256, 96);
    ctx.fillStyle = '#f59e0b';
    ctx.font = '600 44px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('1 kV', 128, 64);
    const displayTexture = new THREE.CanvasTexture(cv);
    displayTexture.colorSpace = THREE.SRGBColorSpace;
    const display = new THREE.Mesh(
      new THREE.PlaneGeometry(1.4, 0.52),
      new THREE.MeshBasicMaterial({ map: displayTexture, toneMapped: false }),
    );
    display.position.set(ampX - 0.6, TABLE_TOP + 1.0, front + 0.005);
    const knob = new THREE.Mesh(
      new THREE.CylinderGeometry(0.16, 0.16, 0.12, 24),
      new THREE.MeshStandardMaterial({ color: '#d1d5db', roughness: 0.3, metalness: 0.9 }),
    );
    knob.rotation.x = Math.PI / 2;
    knob.position.set(ampX - 0.6, TABLE_TOP + 0.4, front + 0.06);
    this.scene.add(amp, display, knob);

    const terminal = (x, color) => {
      const post = new THREE.Mesh(
        new THREE.CylinderGeometry(0.09, 0.09, 0.18, 20),
        new THREE.MeshStandardMaterial({ color, roughness: 0.4, metalness: 0.2 }),
      );
      post.rotation.x = Math.PI / 2;
      post.position.set(x, TABLE_TOP + 0.55, front + 0.09);
      this.scene.add(post);
      return new THREE.Vector3(x, TABLE_TOP + 0.55, front + 0.18);
    };
    const hot = terminal(ampX + 0.6, '#dc2626');
    const ground = terminal(ampX + 1.2, '#1f2937');

    const wire = (points, color) => {
      const curve = new THREE.CatmullRomCurve3(points);
      this.scene.add(new THREE.Mesh(
        new THREE.TubeGeometry(curve, 80, 0.035, 8, false),
        new THREE.MeshStandardMaterial({ color, roughness: 0.5, metalness: 0.1 }),
      ));
    };
    wire([
      new THREE.Vector3(cornerX, CHIP_TOP + 0.03, edgeZ + 0.13),
      new THREE.Vector3(cornerX, CHIP_TOP + 0.6, edgeZ - 0.2),
      new THREE.Vector3(cornerX - 1.2, CHIP_TOP + 0.3, edgeZ - 1.4),
      new THREE.Vector3(hot.x + 1.8, TABLE_TOP + 0.15, hot.z + 0.6),
      hot,
    ], '#dc2626');
    wire([
      new THREE.Vector3(cornerX, CHIP_TOP - CHIP_T - 0.018, edgeZ - 0.4),
      new THREE.Vector3(cornerX - 0.1, CHIP_TOP - CHIP_T - 0.3, edgeZ - 0.9),
      new THREE.Vector3(cornerX - 1.4, TABLE_TOP + 0.1, edgeZ - 1.6),
      new THREE.Vector3(ground.x + 1.0, TABLE_TOP + 0.1, ground.z + 0.4),
      ground,
    ], '#4b5563');

    this._label('Top & bottom electrode contacts', cornerX - 0.4, CHIP_TOP + 0.9, edgeZ - 1.0);
    this._label('High-voltage amplifier', ampX, TABLE_TOP + 1.9, ampZ);
  }

  // ------------------------------------------------------------------ fibers
  _makePulseTexture() {
    // One Gaussian light pulse per texture repeat, along the fiber (u)
    const cv = document.createElement('canvas');
    cv.width = 256;
    cv.height = 4;
    const ctx = cv.getContext('2d');
    for (let i = 0; i < 256; i++) {
      const v = Math.exp(-(((i - 128) / 18) ** 2));
      const c = Math.round(255 * (0.06 + 0.94 * v));
      ctx.fillStyle = `rgb(${c},${c},${c})`;
      ctx.fillRect(i, 0, 1, 4);
    }
    const texture = new THREE.CanvasTexture(cv);
    texture.wrapS = THREE.RepeatWrapping;
    return texture;
  }

  /** A fiber along `points` (light runs from the first to the last point). */
  _makeFiber(points, coreProfile, jacketProfile) {
    const curve = new THREE.CatmullRomCurve3(points);
    const core = new THREE.Mesh(
      sweepGeometry(curve, coreProfile, 160),
      new THREE.MeshStandardMaterial({
        color: '#000000', emissive: this.palette.light, emissiveIntensity: FIBER_FLOOR, side: THREE.DoubleSide,
      }),
    );
    const jacket = new THREE.Mesh(
      sweepGeometry(curve, jacketProfile, 160),
      new THREE.MeshPhysicalMaterial({
        color: '#cbd5e1', roughness: 0.15, transmission: 0.8, thickness: 0.1,
        transparent: true, opacity: 0.35, depthWrite: false, side: THREE.DoubleSide,
      }),
    );
    this.fiberGroup.add(core, jacket);
    // Pulses move at about the speed they cross the chip in the 2D view
    return {
      core, pulseMap: this.pulseTexture.clone(), repeat: curve.getLength() / CHIP_L, end: points[points.length - 1],
    };
  }

  /** Fibers at one facet (side -1: input, +1: output) for ports [[center, width], ...] in um. */
  _makeFiberArray(ports, side, rect, umToScene) {
    const xFacet = (side * CHIP_L) / 2;
    const zs = ports.map(([x]) => x * umToScene);
    // Cores scale with the mode size (x1.5 for visibility); jackets as thick as the pitch allows
    let pitch = Infinity;
    for (let i = 1; i < zs.length; i++) pitch = Math.min(pitch, Math.abs(zs[i] - zs[i - 1]));
    const fibers = ports.map(([, width], i) => {
      let coreProfile;
      let jacketProfile;
      if (rect) {
        // Rectangular (slab-like) multimode fiber matched to the waveguide core width
        coreProfile = rectangle(0.5 * width * umToScene, 0.03);
        jacketProfile = rectangle(0.5 * width * umToScene + 0.05, 0.08);
      } else {
        const core = Math.max(0.02, 0.75 * width * umToScene);
        coreProfile = circle(core);
        jacketProfile = circle(Math.min(core + 0.06, Math.max(core + 0.015, 0.45 * pitch)));
      }
      const spread = ports.length > 1 ? i / (ports.length - 1) - 0.5 : 0;
      const z = zs[i];
      const points = [
        new THREE.Vector3(xFacet + side * 0.02, CHIP_TOP - 0.06, z),
        new THREE.Vector3(xFacet + side * 1.4, CHIP_TOP - 0.06, z),
        new THREE.Vector3(xFacet + side * 3.6, CHIP_TOP - 0.5, z + spread * 2.5),
        new THREE.Vector3(xFacet + side * 6.5, -1.55, z + spread * 4.5),
        new THREE.Vector3(xFacet + side * 9.0, -1.6, z + spread * 5.5),
      ];
      if (side < 0) points.reverse(); // u runs along the light: far end -> chip -> far end
      return this._makeFiber(points, coreProfile, jacketProfile);
    });

    // Fiber-array (V-groove) block holding the fibers at the facet
    const width = Math.max(0.8, Math.max(...zs) - Math.min(...zs) + 0.6);
    const block = new THREE.Mesh(
      new THREE.BoxGeometry(1.2, 0.5, width),
      new THREE.MeshPhysicalMaterial({
        color: '#e2e8f0', roughness: 0.1, transmission: 0.6, thickness: 0.4, transparent: true, opacity: 0.55,
      }),
    );
    block.position.set(side * (CHIP_L / 2 + 0.65), CHIP_TOP - 0.2, (Math.max(...zs) + Math.min(...zs)) / 2);
    this.fiberGroup.add(block);
    return fibers;
  }

  _buildPorts(app) {
    this.fiberGroup.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose();
    });
    this.fiberGroup.clear(); // also removes the labels' DOM elements
    this.inputFibers = [];
    this.outputFibers = [];
    this.sources = [];
    this.bars = [];
    this.digitSprite = null;
    this.digitShown = null;
    this.pulsedShown = null;

    const d = app.demo;
    const ports = d.ports;
    const umToScene = CHIP_W / app.currentLx;
    const shape = ports.in_shape || 'round';
    const labelY = -0.6;
    const labelZ = CHIP_W / 2 + 1.6;

    if (shape === 'sources') {
      // MNIST: light sources integrated at the input facet, one per input beam (pixel)
      const [xMin, xMax] = d.mnist_input_x;
      const count = d.mnist_pixels[0].length;
      const pitch = ((xMax - xMin) / (count - 1)) * umToScene;
      const base = new THREE.Mesh(
        new THREE.BoxGeometry(0.3, 0.02, (xMax - xMin) * umToScene + 2 * pitch),
        new THREE.MeshStandardMaterial({ color: '#111827', roughness: 0.4, metalness: 0.5 }),
      );
      base.position.set(-CHIP_L / 2 + 0.15, CHIP_TOP + 0.01, ((xMin + xMax) / 2) * umToScene);
      this.fiberGroup.add(base);
      const sourceGeometry = new THREE.BoxGeometry(0.2, 0.025, 0.75 * pitch);
      for (let k = 0; k < count; k++) {
        const source = new THREE.Mesh(sourceGeometry, new THREE.MeshBasicMaterial({ color: '#000000', toneMapped: false }));
        source.position.set(-CHIP_L / 2 + 0.15, CHIP_TOP + 0.03, (xMin + (k * (xMax - xMin)) / (count - 1)) * umToScene);
        this.fiberGroup.add(source);
        this.sources.push(source);
      }
      this._label(`${count} integrated light sources (7×7 pixels)`, -CHIP_L / 2 - 1.5, labelY, labelZ, this.fiberGroup);
    } else if (ports.in.length) {
      this.inputFibers = this._makeFiberArray(ports.in, -1, shape === 'rect', umToScene);
      // Share of each input's power launched through each fiber: its power nearest that fiber
      const centers = ports.in.map(([x]) => x);
      this.inputShares = d.intensity_in.map((row) => {
        const share = centers.map(() => 0);
        row.forEach((p, i) => {
          const x = ((i + 0.5) / row.length - 0.5) * app.currentLx;
          let nearest = 0;
          centers.forEach((c, f) => {
            if (Math.abs(x - c) < Math.abs(x - centers[nearest])) nearest = f;
          });
          share[nearest] += p;
        });
        const total = share.reduce((a, b) => a + b, 0) || 1;
        return share.map((p) => p / total);
      });
      const kind = shape === 'rect' ? 'Rectangular multimode input fiber' : `Input fiber${ports.in.length > 1 ? 's' : ''}`;
      this._label(kind, -CHIP_L / 2 - 4.0, labelY, labelZ, this.fiberGroup);
    }

    if (ports.out.length) {
      this.outputFibers = this._makeFiberArray(ports.out, +1, false, umToScene);
      const kind = d.key === 'multimode' ? 'Single-mode output fibers (one per mode)' : `Output fiber${ports.out.length > 1 ? 's' : ''}`;
      this._label(kind, CHIP_L / 2 + 4.0, labelY, labelZ, this.fiberGroup);
    }

    if (shape === 'sources') {
      // MNIST: a bar at the end of each output fiber shows the energy in that digit bin
      this.bars = this.outputFibers.map((f, i) => {
        const bar = new THREE.Mesh(
          new THREE.BoxGeometry(0.4, 1, 0.4),
          // Lit and shaded like the other hardware, with a faint glow
          new THREE.MeshPhysicalMaterial({
            color: this.palette.bar, roughness: 0.3, metalness: 0.2, clearcoat: 0.6, emissive: this.palette.light,
            emissiveIntensity: 0.15,
          }),
        );
        bar.position.set(f.end.x + 0.5, TABLE_TOP, f.end.z);
        this.fiberGroup.add(bar);
        // Digit label riding on top of its bar
        bar.label = this._label(d.detector.labels[i], f.end.x + 0.5, TABLE_TOP, f.end.z, this.fiberGroup);
        bar.label.element.classList.add('small');
        return bar;
      });

      // The input digit, always facing the camera, above the light sources
      const cv = document.createElement('canvas');
      cv.width = 28;
      cv.height = 28;
      this.digitCanvas = cv;
      const texture = new THREE.CanvasTexture(cv);
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.magFilter = THREE.NearestFilter;
      this.digitSprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, toneMapped: false }));
      this.digitSprite.scale.set(1.8, 1.8, 1);
      this.digitSprite.position.set(-CHIP_L / 2 - 2.6, TABLE_TOP + 0.95, 0); // standing on the table
      this.fiberGroup.add(this.digitSprite);
      this._label('Input image', -CHIP_L / 2 - 2.6, TABLE_TOP + 2.1, 0, this.fiberGroup);
    }
  }

  _updatePorts(app) {
    const d = app.demo;
    const r = app.renderer;
    const pulsed = r.pulsed;
    if (pulsed !== this.pulsedShown) {
      // Pulsed light: moving pulses; CW light: steady glow
      for (const f of [...this.inputFibers, ...this.outputFibers]) {
        f.core.material.emissiveMap = pulsed ? f.pulseMap : null;
        f.core.material.needsUpdate = true;
        f.pulseMap.repeat.set(f.repeat, 1);
      }
      this.pulsedShown = pulsed;
    }

    // Phase of the pulse train: pulse k enters at t = k * period, leaves at + transit
    const t = r._pulseTime();
    const fracIn = (t / r.pulsePeriodSec) % 1;
    const fracOut = ((((t - r.pulseTransitSec) / r.pulsePeriodSec) % 1) + 1) % 1;

    // Inputs: the input of the pulse entering the chip (CW: mean over the inputs, like the
    // readout panel, or the selected input in manual mode)
    const mean = (rows) => rows[0].map((_, i) => rows.reduce((s, row) => s + row[i], 0) / rows.length);
    const inL = app.inLayer;
    if (this.inputFibers.length) {
      const shares = inL >= 0 ? this.inputShares[inL] : mean(this.inputShares);
      this.inputFibers.forEach((f, i) => {
        f.core.material.emissiveIntensity = FIBER_FLOOR + FIBER_GAIN * shares[i];
        f.pulseMap.offset.x = 0.5 - f.repeat - fracIn; // a pulse reaches the chip (u = 1) at fracIn = 0
      });
    }
    if (this.sources.length) {
      const pixels = inL >= 0 ? d.mnist_pixels[inL] : mean(d.mnist_pixels);
      // Pulsed light: the sources flash as each pulse is launched (same width as on the chip)
      const phase = Math.min(fracIn, 1 - fracIn) * r.pulsePeriodSec;
      const flash = pulsed ? Math.exp(-4 * Math.LN2 * (phase / r.pulseFwhmSec) ** 2) : 1;
      this.sources.forEach((source, k) => {
        source.material.color.copy(this.palette.light).multiplyScalar(0.04 + 0.96 * pixels[k] * flash);
      });
    }

    if (this.digitSprite && inL !== this.digitShown) {
      // The digit entering the chip (CW, all digits: their mean)
      const images = d.mnist_images;
      const image = inL >= 0 ? images[inL] : mean(images);
      const ctx = this.digitCanvas.getContext('2d');
      const pixels = ctx.createImageData(28, 28);
      image.forEach((v, k) => {
        const c = Math.round(255 * Math.min(1, Math.max(0, v)));
        pixels.data.set([c, c, c, 255], 4 * k);
      });
      ctx.putImageData(pixels, 0, 0);
      this.digitSprite.material.map.needsUpdate = true;
      this.digitShown = inL;
    }

    // Outputs: power coupled into each output fiber's mode, for the pulse at the output
    if (this.outputFibers.length && d.detector && d.detector.powers) {
      const powers = app._outputPowers();
      this.outputFibers.forEach((f, i) => {
        f.core.material.emissiveIntensity = FIBER_FLOOR + FIBER_GAIN * powers[i];
        f.pulseMap.offset.x = 0.5 - fracOut; // a pulse leaves the chip (u = 0) at fracOut = 0
      });
      // MNIST bars: share of the detected light per bin; the prediction (brightest bin) in gold
      const best = powers.indexOf(Math.max(...powers));
      this.bars.forEach((bar, i) => {
        const height = 0.02 + BAR_HEIGHT * powers[i];
        bar.scale.y = height;
        bar.position.y = TABLE_TOP + height / 2;
        bar.label.position.y = TABLE_TOP + height + 0.3;
        const winner = app.outLayer >= 0 && i === best;
        bar.material.color.copy(winner ? this.palette.winner : this.palette.bar);
        bar.material.emissive.copy(winner ? this.palette.winnerGlow : this.palette.light);
      });
    }
  }

  // ------------------------------------------------------------------ per frame
  _resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.labelRenderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** Call right after the 2D renderer has drawn, so its canvas still holds the frame. */
  render(app) {
    if (app.demo && app.demo.ports) {
      const key = `${app.demo.key}|${app.currentLx}|${JSON.stringify(app.demo.ports)}`;
      if (key !== this.fiberKey) {
        this.fiberKey = key;
        this._buildPorts(app);
      }
      this._updatePorts(app);
    }
    this.chipTexture.needsUpdate = true;
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.labelRenderer.render(this.scene, this.camera);
  }
}

// ------------------------------------------------------------------ geometry helpers
function circle(radius, count = 12) {
  return Array.from({ length: count }, (_, j) => {
    const a = (2 * Math.PI * j) / count;
    return [radius * Math.cos(a), radius * Math.sin(a)];
  });
}

function rectangle(halfWidth, halfHeight) {
  return [[-halfWidth, -halfHeight], [halfWidth, -halfHeight], [halfWidth, halfHeight], [-halfWidth, halfHeight]];
}

/**
 * Sweep a closed cross-section (points [side, up]) along a curve. The frame keeps "up" near
 * world +y, so a rectangle stays flat like the chip; u runs along the curve (pulse texture).
 */
function sweepGeometry(curve, profile, segments) {
  const positions = [];
  const uvs = [];
  const indices = [];
  const worldUp = new THREE.Vector3(0, 1, 0);
  const n = profile.length;
  for (let i = 0; i <= segments; i++) {
    const u = i / segments;
    const p = curve.getPointAt(u);
    const tangent = curve.getTangentAt(u);
    const side = new THREE.Vector3().crossVectors(tangent, worldUp).normalize();
    const up = new THREE.Vector3().crossVectors(side, tangent);
    for (let j = 0; j <= n; j++) {
      const [a, b] = profile[j % n];
      positions.push(p.x + side.x * a + up.x * b, p.y + side.y * a + up.y * b, p.z + side.z * a + up.z * b);
      uvs.push(u, j / n);
    }
  }
  for (let i = 0; i < segments; i++) {
    for (let j = 0; j < n; j++) {
      const a = i * (n + 1) + j;
      const b = a + n + 1;
      indices.push(a, b, a + 1, b, b + 1, a + 1);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

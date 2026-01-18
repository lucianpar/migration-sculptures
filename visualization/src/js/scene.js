/**
 * Scene Manager
 *
 * Handles Three.js scene setup, camera, lighting, and render loop.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export class SceneManager {
  constructor(canvas, config) {
    this.canvas = canvas;
    this.config = config;

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;

    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();

    // Callbacks
    this.onObjectClick = null;
    this.onObjectHover = null;
  }

  /**
   * Initialize the scene
   */
  setup() {
    this.createScene();
    this.createCamera();
    this.createRenderer();
    this.createLighting();
    this.createControls();
    this.createGround();
    this.createBackground();

    // Event listeners
    window.addEventListener("resize", () => this.onResize());
    this.canvas.addEventListener("click", (e) => this.onClick(e));
    this.canvas.addEventListener("mousemove", (e) => this.onMouseMove(e));
  }

  /**
   * Create Three.js scene
   */
  createScene() {
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0x0a1628, 20, 80);
  }

  /**
   * Create camera
   */
  createCamera() {
    const aspect = window.innerWidth / window.innerHeight;
    const { fov, near, far, position } = this.config.camera;

    this.camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
    this.camera.position.set(position.x, position.y, position.z);
    this.camera.lookAt(0, 0, 0);
  }

  /**
   * Create WebGL renderer
   */
  createRenderer() {
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: true,
    });

    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
  }

  /**
   * Create scene lighting
   */
  createLighting() {
    // Ambient light for soft overall illumination
    const ambient = new THREE.AmbientLight(0x4a6fa5, 0.4);
    this.scene.add(ambient);

    // Main directional light (sun-like)
    const mainLight = new THREE.DirectionalLight(0xffffff, 1.0);
    mainLight.position.set(10, 20, 10);
    mainLight.castShadow = false;
    this.scene.add(mainLight);

    // Fill light from opposite side
    const fillLight = new THREE.DirectionalLight(0x87ceeb, 0.3);
    fillLight.position.set(-10, 10, -10);
    this.scene.add(fillLight);

    // Rim light for edge definition
    const rimLight = new THREE.DirectionalLight(0x5bc0de, 0.2);
    rimLight.position.set(0, -5, 15);
    this.scene.add(rimLight);

    // Point light for subtle highlights
    const pointLight = new THREE.PointLight(0x4a90d9, 0.5, 30);
    pointLight.position.set(0, 8, 0);
    this.scene.add(pointLight);
  }

  /**
   * Create orbit controls
   */
  createControls() {
    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.minDistance = 5;
    this.controls.maxDistance = 50;
    this.controls.maxPolarAngle = Math.PI / 2.1;
    this.controls.target.set(0, 1, 0);
  }

  /**
   * Create ground plane
   */
  createGround() {
    // Ground plane with subtle reflection
    const groundGeometry = new THREE.PlaneGeometry(100, 100);
    const groundMaterial = new THREE.MeshStandardMaterial({
      color: 0x0a1628,
      metalness: 0.3,
      roughness: 0.8,
    });

    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.5;
    ground.receiveShadow = true;
    ground.name = "ground";
    this.scene.add(ground);

    // Grid helper for orientation
    const gridHelper = new THREE.GridHelper(30, 30, 0x1a3a5c, 0x0d1f33);
    gridHelper.position.y = -0.49;
    this.scene.add(gridHelper);
  }

  /**
   * Create gradient background
   */
  createBackground() {
    // Create gradient background using a large sphere
    const bgGeometry = new THREE.SphereGeometry(80, 32, 32);
    const bgMaterial = new THREE.ShaderMaterial({
      side: THREE.BackSide,
      uniforms: {
        topColor: { value: new THREE.Color(0x0d2140) },
        bottomColor: { value: new THREE.Color(0x0a1628) },
      },
      vertexShader: `
                varying vec3 vWorldPosition;
                void main() {
                    vec4 worldPosition = modelMatrix * vec4(position, 1.0);
                    vWorldPosition = worldPosition.xyz;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
      fragmentShader: `
                uniform vec3 topColor;
                uniform vec3 bottomColor;
                varying vec3 vWorldPosition;
                void main() {
                    float h = normalize(vWorldPosition).y;
                    gl_FragColor = vec4(mix(bottomColor, topColor, max(h, 0.0)), 1.0);
                }
            `,
    });

    const bgSphere = new THREE.Mesh(bgGeometry, bgMaterial);
    this.scene.add(bgSphere);
  }

  /**
   * Handle window resize
   */
  onResize() {
    const width = window.innerWidth;
    const height = window.innerHeight;

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();

    this.renderer.setSize(width, height);
  }

  /**
   * Handle mouse click for object selection
   */
  onClick(event) {
    this.updateMousePosition(event);

    this.raycaster.setFromCamera(this.mouse, this.camera);
    const intersects = this.raycaster.intersectObjects(
      this.scene.children,
      true,
    );

    // Find first sculpture (not ground or background)
    for (const hit of intersects) {
      let object = hit.object;

      // Traverse up to find sculpture group
      while (object.parent && !object.userData.specimen) {
        object = object.parent;
      }

      if (object.userData.specimen && this.onObjectClick) {
        this.onObjectClick(object);
        return;
      }
    }

    // Clicked on nothing - deselect
    if (this.onObjectClick) {
      this.onObjectClick(null);
    }
  }

  /**
   * Handle mouse move for hover effects
   */
  onMouseMove(event) {
    this.updateMousePosition(event);

    // Can implement hover highlighting here
    if (this.onObjectHover) {
      this.raycaster.setFromCamera(this.mouse, this.camera);
      const intersects = this.raycaster.intersectObjects(
        this.scene.children,
        true,
      );

      for (const hit of intersects) {
        let object = hit.object;
        while (object.parent && !object.userData.specimen) {
          object = object.parent;
        }

        if (object.userData.specimen) {
          this.onObjectHover(object);
          return;
        }
      }

      this.onObjectHover(null);
    }
  }

  /**
   * Update mouse position for raycasting
   */
  updateMousePosition(event) {
    const rect = this.canvas.getBoundingClientRect();
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  }

  /**
   * Add object to scene
   */
  add(object) {
    this.scene.add(object);
  }

  /**
   * Remove object from scene
   */
  remove(object) {
    this.scene.remove(object);
  }

  /**
   * Animation loop
   */
  animate() {
    requestAnimationFrame(() => this.animate());

    // Update controls
    this.controls.update();

    // Render
    this.renderer.render(this.scene, this.camera);
  }
}

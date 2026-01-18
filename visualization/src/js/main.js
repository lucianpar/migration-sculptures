/**
 * Migration Sculptures - Main Entry Point
 *
 * Interactive 3D visualization of whale migration sculptures
 * using Three.js.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { SceneManager } from "./scene.js";
import { SculptureManager } from "./sculptures.js";
import { UIManager } from "./ui.js";

// Configuration
const CONFIG = {
  // Grid layout
  grid: {
    rows: 2, // Spring / Fall
    cols: 5, // Years
    spacing: 4, // Spacing between sculptures
  },

  // Camera
  camera: {
    fov: 45,
    near: 0.1,
    far: 1000,
    position: { x: 10, y: 15, z: 20 },
  },

  // Models directory
  modelsPath: "/models/",

  // Initial specimens to load
  specimens: [
    { species: "blue_whale", season: "spring", year: 2010 },
    { species: "blue_whale", season: "spring", year: 2012 },
    { species: "blue_whale", season: "spring", year: 2014 },
    { species: "blue_whale", season: "spring", year: 2016 },
    { species: "blue_whale", season: "spring", year: 2018 },
    { species: "blue_whale", season: "fall", year: 2010 },
    { species: "blue_whale", season: "fall", year: 2012 },
    { species: "blue_whale", season: "fall", year: 2014 },
    { species: "blue_whale", season: "fall", year: 2016 },
    { species: "blue_whale", season: "fall", year: 2018 },
  ],
};

// Application state
let sceneManager;
let sculptureManager;
let uiManager;

/**
 * Initialize the application
 */
async function init() {
  console.log("Migration Sculptures - Initializing...");

  // Get canvas element
  const canvas = document.getElementById("three-canvas");

  // Initialize managers
  sceneManager = new SceneManager(canvas, CONFIG);
  sculptureManager = new SculptureManager(sceneManager, CONFIG);
  uiManager = new UIManager(sculptureManager);

  // Setup scene
  sceneManager.setup();

  // Load sculptures
  try {
    await sculptureManager.loadSpecimens(CONFIG.specimens);
    hideLoading();
  } catch (error) {
    console.warn("Could not load models, showing demo scene:", error);
    sculptureManager.createDemoSculptures();
    hideLoading();
  }

  // Connect UI events
  uiManager.setup();

  // Start render loop
  sceneManager.animate();

  console.log("Migration Sculptures - Ready");
}

/**
 * Hide loading indicator
 */
function hideLoading() {
  const loading = document.getElementById("loading");
  if (loading) {
    loading.classList.add("hidden");
  }
}

// Start application when DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}

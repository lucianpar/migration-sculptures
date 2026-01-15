/**
 * Sculpture Manager
 * 
 * Handles loading, positioning, and managing sculpture models.
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// Species colors
const SPECIES_COLORS = {
    blue_whale: 0x4a90d9,
    fin_whale: 0x7c7c7c,
    gray_whale: 0x6b8e8e,
    humpback_whale: 0x8b6914
};

// Season indicators
const SEASON_INDICATORS = {
    spring: { offset: -0.5, symbol: '↑' },
    fall: { offset: 0.5, symbol: '↓' }
};

export class SculptureManager {
    constructor(sceneManager, config) {
        this.sceneManager = sceneManager;
        this.config = config;
        this.loader = new GLTFLoader();
        
        // Storage
        this.sculptures = new Map();  // id -> sculpture group
        this.specimens = [];           // specimen data
        
        // Selection
        this.selectedSculpture = null;
        
        // Filters
        this.filters = {
            species: 'all',
            season: 'all'
        };
    }
    
    /**
     * Load specimens from model files
     */
    async loadSpecimens(specimens) {
        this.specimens = specimens;
        
        const loadPromises = specimens.map((spec, index) => 
            this.loadSculpture(spec, index)
        );
        
        const results = await Promise.allSettled(loadPromises);
        
        // Log results
        const loaded = results.filter(r => r.status === 'fulfilled').length;
        console.log(`Loaded ${loaded}/${specimens.length} sculptures`);
        
        return results;
    }
    
    /**
     * Load a single sculpture model
     */
    async loadSculpture(specimen, index) {
        const { species, season, year } = specimen;
        const id = `${species}_${year}_${season}`;
        
        // Calculate grid position
        const position = this.calculateGridPosition(index);
        
        // Try to load the actual model
        const modelPath = `${this.config.modelsPath}${id}.glb`;
        
        try {
            const gltf = await this.loadModel(modelPath);
            
            // Create sculpture group
            const group = this.createSculptureGroup(
                gltf.scene,
                specimen,
                position,
                gltf.userData?.extras || {}
            );
            
            this.sculptures.set(id, group);
            this.sceneManager.add(group);
            
            return group;
        } catch (error) {
            console.warn(`Could not load ${modelPath}, creating placeholder`);
            
            // Create placeholder
            const group = this.createPlaceholderSculpture(specimen, position);
            this.sculptures.set(id, group);
            this.sceneManager.add(group);
            
            return group;
        }
    }
    
    /**
     * Load a GLTF model
     */
    loadModel(path) {
        return new Promise((resolve, reject) => {
            this.loader.load(
                path,
                (gltf) => resolve(gltf),
                undefined,
                (error) => reject(error)
            );
        });
    }
    
    /**
     * Calculate grid position for a sculpture
     */
    calculateGridPosition(index) {
        const { rows, cols, spacing } = this.config.grid;
        
        const col = index % cols;
        const row = Math.floor(index / cols);
        
        // Center the grid
        const xOffset = -(cols - 1) * spacing / 2;
        const zOffset = -(rows - 1) * spacing / 2;
        
        return {
            x: col * spacing + xOffset,
            y: 0,
            z: row * spacing + zOffset
        };
    }
    
    /**
     * Create a sculpture group with base and labels
     */
    createSculptureGroup(mesh, specimen, position, metadata) {
        const group = new THREE.Group();
        group.position.set(position.x, position.y, position.z);
        
        // Store specimen data
        group.userData.specimen = {
            ...specimen,
            ...metadata
        };
        
        // Add the main mesh
        if (mesh) {
            // Apply custom material
            mesh.traverse((child) => {
                if (child.isMesh) {
                    child.material = this.createSculptureMaterial(specimen.species);
                }
            });
            
            // Center and scale
            const box = new THREE.Box3().setFromObject(mesh);
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            
            if (maxDim > 0) {
                const scale = 2.0 / maxDim;
                mesh.scale.setScalar(scale);
            }
            
            const center = box.getCenter(new THREE.Vector3());
            mesh.position.sub(center.multiplyScalar(mesh.scale.x));
            mesh.position.y += 1.5;  // Lift above base
            
            group.add(mesh);
        }
        
        // Add base plinth
        const base = this.createBase(specimen);
        group.add(base);
        
        // Add year label
        const label = this.createLabel(specimen.year.toString());
        label.position.y = -0.4;
        label.position.z = 0.8;
        group.add(label);
        
        return group;
    }
    
    /**
     * Create placeholder sculpture (when model not available)
     */
    createPlaceholderSculpture(specimen, position) {
        const group = new THREE.Group();
        group.position.set(position.x, position.y, position.z);
        
        // Store specimen data (with placeholder metrics)
        group.userData.specimen = {
            ...specimen,
            n_tracks: Math.floor(Math.random() * 20) + 10,
            metrics: {
                coherence: { value: Math.random() * 0.5 + 0.4, rating: 'Moderate' },
                entropy: { value: Math.random() * 0.4 + 0.2, rating: 'Moderate' },
                centroid_drift: { value_km: Math.random() * 10, direction: 'North' },
                temporal_variability: { value: Math.random() * 0.4 + 0.2, rating: 'Low' }
            }
        };
        
        // Create procedural sculpture shape
        const sculptureGeometry = this.createProceduralSculptureGeometry(specimen);
        const sculptureMaterial = this.createSculptureMaterial(specimen.species);
        const sculpture = new THREE.Mesh(sculptureGeometry, sculptureMaterial);
        sculpture.position.y = 1.5;
        sculpture.rotation.y = Math.random() * Math.PI * 2;
        group.add(sculpture);
        
        // Add base
        const base = this.createBase(specimen);
        group.add(base);
        
        // Add label
        const label = this.createLabel(specimen.year.toString());
        label.position.y = -0.4;
        label.position.z = 0.8;
        group.add(label);
        
        return group;
    }
    
    /**
     * Create procedural geometry that resembles bundled tracks
     */
    createProceduralSculptureGeometry(specimen) {
        // Create a parametric "bundle" shape
        const segments = 32;
        const tubeRadius = 0.3;
        const height = 2;
        
        // Base shape - twisted tube bundle
        const points = [];
        for (let i = 0; i <= segments; i++) {
            const t = i / segments;
            const twist = Math.sin(t * Math.PI) * 0.5;
            const x = Math.sin(t * Math.PI * 2) * 0.3 + twist;
            const y = t * height - height / 2;
            const z = Math.cos(t * Math.PI * 2) * 0.3;
            points.push(new THREE.Vector3(x, y, z));
        }
        
        const curve = new THREE.CatmullRomCurve3(points);
        const geometry = new THREE.TubeGeometry(curve, segments, tubeRadius, 16, false);
        
        // Add some organic variation
        const positions = geometry.attributes.position;
        for (let i = 0; i < positions.count; i++) {
            const x = positions.getX(i);
            const y = positions.getY(i);
            const z = positions.getZ(i);
            
            const noise = Math.sin(y * 3) * 0.1 + Math.sin(x * 5) * 0.05;
            positions.setX(i, x + noise);
            positions.setZ(i, z + noise);
        }
        
        geometry.computeVertexNormals();
        return geometry;
    }
    
    /**
     * Create sculpture material
     */
    createSculptureMaterial(species) {
        const color = SPECIES_COLORS[species] || 0x4a90d9;
        
        return new THREE.MeshPhysicalMaterial({
            color: color,
            metalness: 0.1,
            roughness: 0.4,
            transmission: 0.3,
            thickness: 1.0,
            transparent: true,
            opacity: 0.85,
            side: THREE.DoubleSide,
            envMapIntensity: 0.5
        });
    }
    
    /**
     * Create base plinth
     */
    createBase(specimen) {
        const color = SPECIES_COLORS[specimen.species] || 0x4a90d9;
        
        // Main base
        const baseGeometry = new THREE.CylinderGeometry(0.8, 1.0, 0.3, 32);
        const baseMaterial = new THREE.MeshStandardMaterial({
            color: color,
            metalness: 0.3,
            roughness: 0.7,
            emissive: color,
            emissiveIntensity: 0.1
        });
        
        const base = new THREE.Mesh(baseGeometry, baseMaterial);
        base.position.y = -0.15;
        
        return base;
    }
    
    /**
     * Create text label (simplified as sprite)
     */
    createLabel(text) {
        // Create canvas texture for text
        const canvas = document.createElement('canvas');
        canvas.width = 128;
        canvas.height = 64;
        const ctx = canvas.getContext('2d');
        
        ctx.fillStyle = '#5bc0de';
        ctx.font = 'bold 32px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, 64, 32);
        
        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({
            map: texture,
            transparent: true
        });
        
        const sprite = new THREE.Sprite(material);
        sprite.scale.set(1, 0.5, 1);
        
        return sprite;
    }
    
    /**
     * Create demo sculptures when models aren't available
     */
    createDemoSculptures() {
        console.log('Creating demo sculptures');
        
        this.config.specimens.forEach((spec, index) => {
            const position = this.calculateGridPosition(index);
            const group = this.createPlaceholderSculpture(spec, position);
            const id = `${spec.species}_${spec.year}_${spec.season}`;
            
            this.sculptures.set(id, group);
            this.sceneManager.add(group);
        });
        
        this.specimens = this.config.specimens;
    }
    
    /**
     * Select a sculpture
     */
    select(sculpture) {
        // Deselect previous
        if (this.selectedSculpture) {
            this.setHighlight(this.selectedSculpture, false);
        }
        
        this.selectedSculpture = sculpture;
        
        if (sculpture) {
            this.setHighlight(sculpture, true);
        }
        
        return sculpture?.userData?.specimen;
    }
    
    /**
     * Set highlight effect on sculpture
     */
    setHighlight(sculpture, highlighted) {
        sculpture.traverse((child) => {
            if (child.isMesh && child.material) {
                if (highlighted) {
                    child.material.emissiveIntensity = 0.3;
                } else {
                    child.material.emissiveIntensity = 0.1;
                }
            }
        });
    }
    
    /**
     * Apply filters to show/hide sculptures
     */
    applyFilters(filters) {
        this.filters = { ...this.filters, ...filters };
        
        this.sculptures.forEach((sculpture, id) => {
            const specimen = sculpture.userData.specimen;
            
            let visible = true;
            
            if (this.filters.species !== 'all' && specimen.species !== this.filters.species) {
                visible = false;
            }
            
            if (this.filters.season !== 'all' && specimen.season !== this.filters.season) {
                visible = false;
            }
            
            sculpture.visible = visible;
        });
    }
    
    /**
     * Get all sculptures
     */
    getAll() {
        return Array.from(this.sculptures.values());
    }
    
    /**
     * Get sculpture by ID
     */
    get(id) {
        return this.sculptures.get(id);
    }
}

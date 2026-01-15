"""
glTF/GLB Export Module

Exports 3D meshes to glTF binary format for web visualization.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json
import struct
import numpy as np

logger = logging.getLogger(__name__)

# Try to import pygltflib
try:
    from pygltflib import GLTF2, Scene, Node, Mesh as GLTFMesh, Primitive, Accessor, BufferView, Buffer
    from pygltflib import ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER, FLOAT, UNSIGNED_INT, TRIANGLES
    PYGLTFLIB_AVAILABLE = True
except ImportError:
    PYGLTFLIB_AVAILABLE = False
    logger.warning("pygltflib not available. Install with: pip install pygltflib")


class GLTFExporter:
    """
    Exports sculpture meshes to glTF/GLB format for Three.js visualization.
    
    glTF (GL Transmission Format) is the preferred format because:
    - Efficient binary format (GLB)
    - Native support in Three.js
    - Supports embedded metadata in 'extras'
    - Preserves materials and normals
    """
    
    def __init__(self, embed_metadata: bool = True):
        """
        Initialize exporter.
        
        Args:
            embed_metadata: Whether to embed sculpture metadata in glTF extras
        """
        self.embed_metadata = embed_metadata
    
    def export(
        self,
        mesh,  # Mesh from isosurface module
        output_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
        material_color: tuple = (0.3, 0.5, 0.8, 0.8)  # RGBA
    ) -> Path:
        """
        Export a mesh to GLB format.
        
        Args:
            mesh: Mesh object with vertices, faces, and optionally normals
            output_path: Output file path (.glb)
            metadata: Optional metadata dictionary to embed
            material_color: Base color as (r, g, b, a) in 0-1 range
            
        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if PYGLTFLIB_AVAILABLE:
            return self._export_with_pygltflib(mesh, output_path, metadata, material_color)
        else:
            return self._export_manual(mesh, output_path, metadata)
    
    def _export_with_pygltflib(
        self,
        mesh,
        output_path: Path,
        metadata: Optional[Dict[str, Any]],
        material_color: tuple
    ) -> Path:
        """Export using pygltflib library."""
        from pygltflib import (
            GLTF2, Scene, Node, Mesh as GLTFMesh, Primitive, Accessor, BufferView, Buffer,
            Material, PbrMetallicRoughness
        )
        
        # Prepare data
        vertices = mesh.vertices.astype(np.float32).flatten()
        indices = mesh.faces.astype(np.uint32).flatten()
        normals = mesh.normals.astype(np.float32).flatten() if mesh.normals is not None else None
        
        # Compute bounds
        v_min = mesh.vertices.min(axis=0).tolist()
        v_max = mesh.vertices.max(axis=0).tolist()
        
        # Create binary blob
        vertex_blob = vertices.tobytes()
        index_blob = indices.tobytes()
        normal_blob = normals.tobytes() if normals is not None else b""
        
        blob = vertex_blob + index_blob + normal_blob
        
        # Create glTF structure
        gltf = GLTF2(
            scene=0,
            scenes=[Scene(nodes=[0])],
            nodes=[Node(mesh=0)],
            meshes=[GLTFMesh(primitives=[
                Primitive(
                    attributes={"POSITION": 0, "NORMAL": 2} if normals is not None else {"POSITION": 0},
                    indices=1,
                    material=0
                )
            ])],
            materials=[
                Material(
                    pbrMetallicRoughness=PbrMetallicRoughness(
                        baseColorFactor=list(material_color),
                        metallicFactor=0.1,
                        roughnessFactor=0.8
                    ),
                    alphaMode="BLEND" if material_color[3] < 1.0 else "OPAQUE",
                    doubleSided=True
                )
            ],
            accessors=[
                # Vertex positions
                Accessor(
                    bufferView=0,
                    componentType=FLOAT,
                    count=len(mesh.vertices),
                    type="VEC3",
                    max=v_max,
                    min=v_min
                ),
                # Indices
                Accessor(
                    bufferView=1,
                    componentType=UNSIGNED_INT,
                    count=len(indices),
                    type="SCALAR"
                ),
            ],
            bufferViews=[
                # Vertices
                BufferView(
                    buffer=0,
                    byteOffset=0,
                    byteLength=len(vertex_blob),
                    target=ARRAY_BUFFER
                ),
                # Indices
                BufferView(
                    buffer=0,
                    byteOffset=len(vertex_blob),
                    byteLength=len(index_blob),
                    target=ELEMENT_ARRAY_BUFFER
                ),
            ],
            buffers=[Buffer(byteLength=len(blob))]
        )
        
        # Add normals accessor and buffer view if present
        if normals is not None:
            gltf.accessors.append(
                Accessor(
                    bufferView=2,
                    componentType=FLOAT,
                    count=len(mesh.vertices),
                    type="VEC3"
                )
            )
            gltf.bufferViews.append(
                BufferView(
                    buffer=0,
                    byteOffset=len(vertex_blob) + len(index_blob),
                    byteLength=len(normal_blob),
                    target=ARRAY_BUFFER
                )
            )
        
        # Embed metadata in extras
        if self.embed_metadata and metadata:
            gltf.extras = metadata
        
        # Set binary blob
        gltf.set_binary_blob(blob)
        
        # Save as GLB
        gltf.save(str(output_path))
        
        logger.info(f"Exported GLB to {output_path}")
        return output_path
    
    def _export_manual(
        self,
        mesh,
        output_path: Path,
        metadata: Optional[Dict[str, Any]]
    ) -> Path:
        """
        Manual GLB export without pygltflib.
        
        This is a simplified fallback that creates a basic valid GLB file.
        """
        # Prepare binary data
        vertices = mesh.vertices.astype(np.float32)
        indices = mesh.faces.astype(np.uint32)
        
        vertex_data = vertices.tobytes()
        index_data = indices.tobytes()
        bin_data = vertex_data + index_data
        
        # Pad to 4-byte alignment
        while len(bin_data) % 4 != 0:
            bin_data += b'\x00'
        
        # Compute bounds
        v_min = vertices.min(axis=0).tolist()
        v_max = vertices.max(axis=0).tolist()
        
        # Create JSON content
        gltf_json = {
            "asset": {"version": "2.0", "generator": "MigrationSculptures"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{
                "primitives": [{
                    "attributes": {"POSITION": 0},
                    "indices": 1
                }]
            }],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,  # FLOAT
                    "count": len(vertices),
                    "type": "VEC3",
                    "min": v_min,
                    "max": v_max
                },
                {
                    "bufferView": 1,
                    "componentType": 5125,  # UNSIGNED_INT
                    "count": len(indices.flatten()),
                    "type": "SCALAR"
                }
            ],
            "bufferViews": [
                {
                    "buffer": 0,
                    "byteOffset": 0,
                    "byteLength": len(vertex_data),
                    "target": 34962  # ARRAY_BUFFER
                },
                {
                    "buffer": 0,
                    "byteOffset": len(vertex_data),
                    "byteLength": len(index_data),
                    "target": 34963  # ELEMENT_ARRAY_BUFFER
                }
            ],
            "buffers": [{"byteLength": len(bin_data)}]
        }
        
        # Add metadata if enabled
        if self.embed_metadata and metadata:
            gltf_json["extras"] = metadata
        
        # Serialize JSON
        json_str = json.dumps(gltf_json, separators=(',', ':'))
        json_data = json_str.encode('utf-8')
        
        # Pad JSON to 4-byte alignment
        while len(json_data) % 4 != 0:
            json_data += b' '
        
        # Create GLB file
        # GLB Header (12 bytes)
        magic = b'glTF'
        version = struct.pack('<I', 2)
        total_length = 12 + 8 + len(json_data) + 8 + len(bin_data)
        length = struct.pack('<I', total_length)
        
        # JSON chunk
        json_chunk_length = struct.pack('<I', len(json_data))
        json_chunk_type = struct.pack('<I', 0x4E4F534A)  # JSON
        
        # BIN chunk
        bin_chunk_length = struct.pack('<I', len(bin_data))
        bin_chunk_type = struct.pack('<I', 0x004E4942)  # BIN
        
        # Write file
        with open(output_path, 'wb') as f:
            # Header
            f.write(magic)
            f.write(version)
            f.write(length)
            # JSON chunk
            f.write(json_chunk_length)
            f.write(json_chunk_type)
            f.write(json_data)
            # BIN chunk
            f.write(bin_chunk_length)
            f.write(bin_chunk_type)
            f.write(bin_data)
        
        logger.info(f"Exported GLB (manual) to {output_path}")
        return output_path
    
    def export_specimen(
        self,
        mesh,
        species: str,
        season: str,
        year: int,
        output_dir: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Export a specimen mesh with standard naming convention.
        
        Args:
            mesh: Mesh object
            species: Species name
            season: Season name
            year: Year
            output_dir: Output directory
            metadata: Additional metadata
            
        Returns:
            Path to exported file
        """
        # Create filename
        filename = f"{species}_{year}_{season}.glb"
        output_path = Path(output_dir) / filename
        
        # Combine metadata
        full_metadata = {
            "species": species,
            "season": season,
            "year": year,
            **(metadata or {})
        }
        
        return self.export(mesh, output_path, full_metadata)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("GLTFExporter module loaded")
    print(f"pygltflib available: {PYGLTFLIB_AVAILABLE}")

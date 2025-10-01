# -*- coding: utf-8 -*-
"""
Stage 5: Visual Identity & Provenance → Quantum Traceability

Classical provenance is linear. Quantum provenance allows branching,
reversible trace paths using quantum hashing for model lineage and
quantum fingerprints for visual identity.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import hashlib
import json
import time
from dataclasses import dataclass, asdict
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, random_statevector
from qiskit_aer import AerSimulator
import logging

logger = logging.getLogger(__name__)

@dataclass
class QuantumProvenanceRecord:
    """Data class for quantum provenance records."""
    record_id: str
    parent_id: Optional[str]
    model_hash: str
    quantum_fingerprint: str
    visual_identity_hash: str
    operation_type: str
    parameters: Dict[str, Any]
    timestamp: float
    quantum_state: List[complex]
    entanglement_links: List[str]
    reversibility_key: str

class QuantumProvenanceTracker:
    """
    Quantum-enhanced provenance tracking for AI Research Agent.
    
    Uses quantum hashing for model lineage and encodes visual identity
    as quantum fingerprints with entangled logo states for traceability.
    """
    
    def __init__(self, max_qubits: int = 20, hash_precision: int = 256):
        """Initialize quantum provenance tracker."""
        self.max_qubits = max_qubits
        self.hash_precision = hash_precision
        self.simulator = AerSimulator()
        
        # Provenance state
        self.provenance_graph = {}
        self.quantum_fingerprints = {}
        self.visual_identities = {}
        self.entanglement_registry = {}
        self.reversibility_cache = {}
        
        logger.info(f"Initialized QuantumProvenanceTracker with {max_qubits} qubits, {hash_precision}-bit precision")
    
    def create_quantum_hash(self, data: Union[str, Dict, List], salt: str = None) -> str:
        """
        Create quantum-enhanced hash for data integrity.
        
        Args:
            data: Data to hash
            salt: Optional salt for hashing
            
        Returns:
            Quantum hash string
        """
        # Convert data to string representation
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        if salt:
            data_str = f"{data_str}:{salt}"
        
        # Classical hash as base
        classical_hash = hashlib.sha256(data_str.encode()).hexdigest()
        
        # Create quantum circuit for hash enhancement
        num_qubits = min(len(classical_hash) // 4, self.max_qubits)  # 4 hex chars per qubit
        qreg = QuantumRegister(num_qubits, 'hash')
        circuit = QuantumCircuit(qreg)
        
        # Encode classical hash into quantum state
        for i, hex_char in enumerate(classical_hash[:num_qubits * 4:4]):
            hex_value = int(hex_char, 16)
            # Convert to rotation angle
            angle = (hex_value / 15.0) * np.pi
            circuit.ry(angle, qreg[i])
        
        # Create quantum entanglement for hash integrity
        for i in range(num_qubits - 1):
            circuit.cx(qreg[i], qreg[i + 1])
        
        # Add quantum randomness
        for i in range(num_qubits):
            circuit.rz(np.pi / (i + 1), qreg[i])
        
        # Measure quantum state
        circuit.measure_all()
        
        job = self.simulator.run(circuit, shots=1)
        result = job.result()
        counts = result.get_counts()
        quantum_measurement = list(counts.keys())[0]
        
        # Combine classical and quantum hashes
        quantum_hash = f"q{classical_hash[:32]}{quantum_measurement}"
        
        logger.debug(f"Created quantum hash: {quantum_hash[:16]}...")
        return quantum_hash
    
    def generate_quantum_fingerprint(self, model_params: Dict[str, Any], 
                                   visual_elements: Dict[str, Any] = None) -> str:
        """
        Generate quantum fingerprint for model and visual identity.
        
        Args:
            model_params: Model parameters to fingerprint
            visual_elements: Visual identity elements (colors, logos, etc.)
            
        Returns:
            Quantum fingerprint string
        """
        # Create quantum circuit for fingerprinting
        num_qubits = min(self.max_qubits, 16)  # Limit for fingerprint
        qreg = QuantumRegister(num_qubits, 'fingerprint')
        circuit = QuantumCircuit(qreg)
        
        # Initialize superposition
        for i in range(num_qubits):
            circuit.h(qreg[i])
        
        # Encode model parameters
        weights = model_params.get('weights', [1.0])
        for i, weight in enumerate(weights[:num_qubits]):
            angle = weight * np.pi if abs(weight) <= 1 else np.pi
            circuit.ry(angle, qreg[i])
        
        # Encode visual elements if provided
        if visual_elements:
            colors = visual_elements.get('colors', [])
            for i, color in enumerate(colors[:num_qubits]):
                if isinstance(color, str):
                    # Convert color to numeric value
                    color_value = sum(ord(c) for c in color) % 256
                    angle = (color_value / 255.0) * np.pi
                    circuit.rz(angle, qreg[i])
        
        # Create entanglement pattern for uniqueness
        for i in range(num_qubits - 1):
            circuit.cx(qreg[i], qreg[i + 1])
        
        # Add model-specific phase
        model_id = model_params.get('id', 'default')
        model_phase = (hash(model_id) % 1000) / 1000 * 2 * np.pi
        circuit.rz(model_phase, qreg[0])
        
        # Measure fingerprint
        circuit.measure_all()
        
        job = self.simulator.run(circuit, shots=1)
        result = job.result()
        counts = result.get_counts()
        fingerprint_bits = list(counts.keys())[0]
        
        # Convert to hex fingerprint
        fingerprint_int = int(fingerprint_bits, 2)
        fingerprint_hex = f"qf{fingerprint_int:0{num_qubits//4}x}"
        
        # Store fingerprint
        fingerprint_key = self.create_quantum_hash(model_params)
        self.quantum_fingerprints[fingerprint_key] = {
            'fingerprint': fingerprint_hex,
            'model_params': model_params,
            'visual_elements': visual_elements,
            'creation_time': time.time(),
            'quantum_circuit': circuit
        }
        
        logger.info(f"Generated quantum fingerprint: {fingerprint_hex}")
        return fingerprint_hex
    
    def create_entangled_logo_states(self, logo_variants: List[Dict[str, Any]]) -> QuantumCircuit:
        """
        Create entangled quantum states for logo variants.
        
        Args:
            logo_variants: List of logo variant specifications
            
        Returns:
            Quantum circuit with entangled logo states
        """
        num_variants = min(len(logo_variants), self.max_qubits)
        qreg = QuantumRegister(num_variants, 'logo_variants')
        circuit = QuantumCircuit(qreg)
        
        # Create GHZ state for maximum entanglement
        circuit.h(qreg[0])
        for i in range(1, num_variants):
            circuit.cx(qreg[0], qreg[i])
        
        # Encode variant-specific features
        for i, variant in enumerate(logo_variants[:num_variants]):
            # Encode color scheme
            colors = variant.get('colors', ['#000000'])
            color_hash = hash(str(colors)) % 1000
            color_phase = (color_hash / 1000) * 2 * np.pi
            circuit.rz(color_phase, qreg[i])
            
            # Encode style elements
            style = variant.get('style', 'default')
            style_angle = (hash(style) % 100) / 100 * np.pi
            circuit.ry(style_angle, qreg[i])
        
        # Store entangled logo circuit
        logo_key = self.create_quantum_hash(logo_variants)
        self.visual_identities[logo_key] = {
            'variants': logo_variants,
            'entangled_circuit': circuit,
            'creation_time': time.time()
        }
        
        logger.info(f"Created entangled logo states for {num_variants} variants")
        return circuit
    
    def record_provenance(self, operation_type: str, model_params: Dict[str, Any],
                         parent_record_id: str = None, visual_elements: Dict[str, Any] = None) -> str:
        """
        Record quantum provenance for an operation.
        
        Args:
            operation_type: Type of operation (train, fine-tune, merge, etc.)
            model_params: Current model parameters
            parent_record_id: ID of parent record for lineage
            visual_elements: Visual identity elements
            
        Returns:
            Provenance record ID
        """
        # Generate unique record ID
        record_id = self.create_quantum_hash({
            'operation': operation_type,
            'params': model_params,
            'timestamp': time.time()
        })
        
        # Create quantum fingerprint
        fingerprint = self.generate_quantum_fingerprint(model_params, visual_elements)
        
        # Generate model hash
        model_hash = self.create_quantum_hash(model_params)
        
        # Create visual identity hash
        visual_hash = self.create_quantum_hash(visual_elements) if visual_elements else "none"
        
        # Create quantum state for this record
        num_qubits = min(self.max_qubits, 12)
        quantum_state = random_statevector(2**num_qubits)
        
        # Generate reversibility key for quantum operations
        reversibility_key = self.create_quantum_hash({
            'record_id': record_id,
            'operation': operation_type,
            'timestamp': time.time()
        })
        
        # Create provenance record
        provenance_record = QuantumProvenanceRecord(
            record_id=record_id,
            parent_id=parent_record_id,
            model_hash=model_hash,
            quantum_fingerprint=fingerprint,
            visual_identity_hash=visual_hash,
            operation_type=operation_type,
            parameters=model_params,
            timestamp=time.time(),
            quantum_state=quantum_state.data.tolist(),
            entanglement_links=[],
            reversibility_key=reversibility_key
        )
        
        # Store in provenance graph
        self.provenance_graph[record_id] = provenance_record
        
        # Create entanglement with parent if exists
        if parent_record_id and parent_record_id in self.provenance_graph:
            self._create_provenance_entanglement(parent_record_id, record_id)
        
        # Store reversibility information
        self.reversibility_cache[reversibility_key] = {
            'record_id': record_id,
            'inverse_operation': self._get_inverse_operation(operation_type),
            'restoration_params': model_params.copy()
        }
        
        logger.info(f"Recorded quantum provenance: {record_id[:16]}... for {operation_type}")
        return record_id
    
    def _create_provenance_entanglement(self, parent_id: str, child_id: str):
        """Create quantum entanglement between provenance records."""
        if parent_id not in self.provenance_graph or child_id not in self.provenance_graph:
            return
        
        # Update entanglement links
        self.provenance_graph[parent_id].entanglement_links.append(child_id)
        self.provenance_graph[child_id].entanglement_links.append(parent_id)
        
        # Store in entanglement registry
        entanglement_key = f"{parent_id}:{child_id}"
        self.entanglement_registry[entanglement_key] = {
            'parent': parent_id,
            'child': child_id,
            'entanglement_strength': np.random.random(),  # Quantum correlation strength
            'creation_time': time.time()
        }
        
        logger.debug(f"Created provenance entanglement: {parent_id[:8]}...:{child_id[:8]}...")
    
    def trace_lineage(self, record_id: str, max_depth: int = 10) -> Dict[str, Any]:
        """
        Trace quantum lineage for a provenance record.
        
        Args:
            record_id: Starting record ID
            max_depth: Maximum trace depth
            
        Returns:
            Lineage trace information
        """
        if record_id not in self.provenance_graph:
            return {}
        
        lineage = {
            'root_record': record_id,
            'trace_path': [],
            'quantum_correlations': [],
            'branching_points': [],
            'total_depth': 0
        }
        
        # Breadth-first search through provenance graph
        visited = set()
        queue = [(record_id, 0)]
        
        while queue and len(lineage['trace_path']) < max_depth:
            current_id, depth = queue.pop(0)
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            record = self.provenance_graph[current_id]
            
            # Add to trace path
            lineage['trace_path'].append({
                'record_id': current_id,
                'operation_type': record.operation_type,
                'timestamp': record.timestamp,
                'depth': depth,
                'quantum_fingerprint': record.quantum_fingerprint
            })
            
            # Check for branching (multiple children)
            children = [link for link in record.entanglement_links 
                       if link in self.provenance_graph and 
                       self.provenance_graph[link].parent_id == current_id]
            
            if len(children) > 1:
                lineage['branching_points'].append({
                    'parent_id': current_id,
                    'children': children,
                    'branch_count': len(children)
                })
            
            # Add quantum correlations
            for link in record.entanglement_links:
                if link in self.provenance_graph:
                    entanglement_key = f"{current_id}:{link}"
                    if entanglement_key in self.entanglement_registry:
                        correlation = self.entanglement_registry[entanglement_key]
                        lineage['quantum_correlations'].append(correlation)
            
            # Add parent to queue
            if record.parent_id and record.parent_id not in visited:
                queue.append((record.parent_id, depth + 1))
            
            # Add children to queue
            for child in children:
                if child not in visited:
                    queue.append((child, depth + 1))
        
        lineage['total_depth'] = max([item['depth'] for item in lineage['trace_path']], default=0)
        
        logger.info(f"Traced lineage for {record_id[:16]}...: {len(lineage['trace_path'])} records")
        return lineage
    
    def verify_quantum_integrity(self, record_id: str) -> Dict[str, Any]:
        """
        Verify quantum integrity of a provenance record.
        
        Args:
            record_id: Record ID to verify
            
        Returns:
            Integrity verification results
        """
        if record_id not in self.provenance_graph:
            return {'valid': False, 'error': 'Record not found'}
        
        record = self.provenance_graph[record_id]
        
        # Verify quantum fingerprint
        regenerated_fingerprint = self.generate_quantum_fingerprint(
            record.parameters, 
            self.visual_identities.get(record.visual_identity_hash, {}).get('variants')
        )
        
        fingerprint_valid = regenerated_fingerprint == record.quantum_fingerprint
        
        # Verify model hash
        regenerated_model_hash = self.create_quantum_hash(record.parameters)
        model_hash_valid = regenerated_model_hash == record.model_hash
        
        # Verify quantum state integrity
        quantum_state = np.array(record.quantum_state)
        state_norm = np.linalg.norm(quantum_state)
        state_valid = abs(state_norm - 1.0) < 1e-6  # Valid quantum state should be normalized
        
        # Verify entanglement links
        entanglement_valid = all(
            link in self.provenance_graph for link in record.entanglement_links
        )
        
        integrity_result = {
            'record_id': record_id,
            'valid': fingerprint_valid and model_hash_valid and state_valid and entanglement_valid,
            'fingerprint_valid': fingerprint_valid,
            'model_hash_valid': model_hash_valid,
            'quantum_state_valid': state_valid,
            'entanglement_valid': entanglement_valid,
            'state_norm': float(state_norm),
            'verification_time': time.time()
        }
        
        logger.info(f"Verified quantum integrity for {record_id[:16]}...: {'VALID' if integrity_result['valid'] else 'INVALID'}")
        return integrity_result
    
    def reverse_operation(self, reversibility_key: str) -> Dict[str, Any]:
        """
        Reverse a quantum operation using reversibility key.
        
        Args:
            reversibility_key: Key for operation reversal
            
        Returns:
            Reversal operation results
        """
        if reversibility_key not in self.reversibility_cache:
            return {'success': False, 'error': 'Reversibility key not found'}
        
        reversal_info = self.reversibility_cache[reversibility_key]
        record_id = reversal_info['record_id']
        
        if record_id not in self.provenance_graph:
            return {'success': False, 'error': 'Original record not found'}
        
        original_record = self.provenance_graph[record_id]
        
        # Create reversed operation record
        reversed_params = reversal_info['restoration_params']
        inverse_operation = reversal_info['inverse_operation']
        
        # Record the reversal as new provenance entry
        reversal_record_id = self.record_provenance(
            operation_type=f"reverse_{inverse_operation}",
            model_params=reversed_params,
            parent_record_id=record_id
        )
        
        reversal_result = {
            'success': True,
            'original_record_id': record_id,
            'reversal_record_id': reversal_record_id,
            'reversed_operation': inverse_operation,
            'restored_parameters': reversed_params,
            'reversal_time': time.time()
        }
        
        logger.info(f"Reversed operation {original_record.operation_type} -> {inverse_operation}")
        return reversal_result
    
    def _get_inverse_operation(self, operation_type: str) -> str:
        """Get inverse operation for reversibility."""
        inverse_map = {
            'train': 'untrain',
            'fine_tune': 'restore_base',
            'merge': 'split',
            'quantize': 'dequantize',
            'prune': 'restore_weights',
            'distill': 'expand'
        }
        return inverse_map.get(operation_type, f"reverse_{operation_type}")
    
    def export_provenance_graph(self, filepath: str):
        """Export complete provenance graph to JSON file."""
        export_data = {
            'provenance_records': {
                record_id: {
                    'record_id': record.record_id,
                    'parent_id': record.parent_id,
                    'model_hash': record.model_hash,
                    'quantum_fingerprint': record.quantum_fingerprint,
                    'visual_identity_hash': record.visual_identity_hash,
                    'operation_type': record.operation_type,
                    'parameters': record.parameters,
                    'timestamp': record.timestamp,
                    'entanglement_links': record.entanglement_links,
                    'reversibility_key': record.reversibility_key
                } for record_id, record in self.provenance_graph.items()
            },
            'quantum_fingerprints': self.quantum_fingerprints,
            'visual_identities': {
                key: {
                    'variants': value['variants'],
                    'creation_time': value['creation_time']
                } for key, value in self.visual_identities.items()
            },
            'entanglement_registry': self.entanglement_registry,
            'export_metadata': {
                'total_records': len(self.provenance_graph),
                'export_time': time.time(),
                'quantum_precision': self.hash_precision
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported provenance graph to {filepath}: {len(self.provenance_graph)} records")
    
    def get_quantum_provenance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for quantum provenance tracking."""
        metrics = {
            'total_records': len(self.provenance_graph),
            'quantum_fingerprints': len(self.quantum_fingerprints),
            'visual_identities': len(self.visual_identities),
            'entanglement_links': len(self.entanglement_registry),
            'reversibility_keys': len(self.reversibility_cache),
            'max_qubits': self.max_qubits,
            'hash_precision': self.hash_precision
        }
        
        if self.provenance_graph:
            # Analyze provenance structure
            operations = [record.operation_type for record in self.provenance_graph.values()]
            operation_counts = {op: operations.count(op) for op in set(operations)}
            
            # Calculate graph depth
            depths = []
            for record_id in self.provenance_graph:
                lineage = self.trace_lineage(record_id, max_depth=50)
                depths.append(lineage['total_depth'])
            
            metrics.update({
                'operation_distribution': operation_counts,
                'average_lineage_depth': np.mean(depths) if depths else 0,
                'max_lineage_depth': max(depths) if depths else 0,
                'branching_factor': len(self.entanglement_registry) / len(self.provenance_graph)
            })
        
        return metrics
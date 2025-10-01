#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum LIMIT-Graph v2.0 Setup Script

Automated setup and configuration for quantum-enhanced AI research agent.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuantumSetup:
    """Setup manager for Quantum LIMIT-Graph v2.0."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.requirements_file = self.project_root / "requirements.txt"
        self.config_dir = self.project_root / "config"
        
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            logger.error("Python 3.8+ is required for Quantum LIMIT-Graph v2.0")
            return False
        
        logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    
    def install_quantum_dependencies(self) -> bool:
        """Install quantum computing dependencies."""
        logger.info("Installing quantum computing dependencies...")
        
        try:
            # Install core quantum packages
            quantum_packages = [
                "qiskit>=0.45.0",
                "qiskit-aer>=0.13.0", 
                "qiskit-algorithms>=0.2.0",
                "pennylane>=0.32.0",
                "cirq-core>=1.2.0",
                "lambeq>=0.3.4"
            ]
            
            for package in quantum_packages:
                logger.info(f"Installing {package}...")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Failed to install {package}: {result.stderr}")
                    return False
                
                logger.info(f"✓ {package} installed successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error installing quantum dependencies: {e}")
            return False
    
    def install_requirements(self) -> bool:
        """Install all requirements from requirements.txt."""
        if not self.requirements_file.exists():
            logger.error(f"Requirements file not found: {self.requirements_file}")
            return False
        
        logger.info("Installing all requirements...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to install requirements: {result.stderr}")
                return False
            
            logger.info("✓ All requirements installed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error installing requirements: {e}")
            return False
    
    def verify_quantum_installation(self) -> bool:
        """Verify quantum computing packages are working."""
        logger.info("Verifying quantum installations...")
        
        # Test Qiskit
        try:
            import qiskit
            from qiskit import QuantumCircuit
            from qiskit_aer import AerSimulator
            
            # Create simple test circuit
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure_all()
            
            # Test simulation
            simulator = AerSimulator()
            job = simulator.run(qc, shots=100)
            result = job.result()
            
            logger.info(f"✓ Qiskit {qiskit.__version__} working correctly")
            
        except Exception as e:
            logger.error(f"Qiskit verification failed: {e}")
            return False
        
        # Test PennyLane
        try:
            import pennylane as qml
            
            # Create simple test device
            dev = qml.device('default.qubit', wires=2)
            
            @qml.qnode(dev)
            def test_circuit():
                qml.Hadamard(wires=0)
                qml.CNOT(wires=[0, 1])
                return qml.expval(qml.PauliZ(0))
            
            result = test_circuit()
            logger.info(f"✓ PennyLane {qml.__version__} working correctly")
            
        except Exception as e:
            logger.error(f"PennyLane verification failed: {e}")
            return False
        
        # Test Cirq
        try:
            import cirq
            
            # Create simple test circuit
            qubit = cirq.GridQubit(0, 0)
            circuit = cirq.Circuit(cirq.H(qubit))
            
            logger.info(f"✓ Cirq {cirq.__version__} working correctly")
            
        except Exception as e:
            logger.error(f"Cirq verification failed: {e}")
            return False
        
        # Test Lambeq
        try:
            import lambeq
            from lambeq import AtomicType
            
            # Test basic functionality
            noun_type = AtomicType.NOUN
            logger.info(f"✓ Lambeq {lambeq.__version__} working correctly")
            
        except Exception as e:
            logger.error(f"Lambeq verification failed: {e}")
            return False
        
        logger.info("✅ All quantum packages verified successfully")
        return True
    
    def create_config_files(self) -> bool:
        """Create configuration files for quantum components."""
        logger.info("Creating configuration files...")
        
        try:
            # Create config directory
            self.config_dir.mkdir(exist_ok=True)
            
            # Quantum configuration
            quantum_config = {
                "quantum_backend": "qiskit_aer",
                "max_qubits": 24,
                "default_languages": ["indonesian", "arabic", "spanish", "english"],
                "components": {
                    "semantic_graph": {
                        "enabled": True,
                        "max_qubits": 20
                    },
                    "policy_optimizer": {
                        "enabled": True,
                        "num_qubits": 16,
                        "num_layers": 3
                    },
                    "context_engine": {
                        "enabled": True,
                        "max_context_qubits": 20,
                        "cultural_dimensions": 8
                    },
                    "benchmark_harness": {
                        "enabled": True,
                        "max_qubits": 24
                    },
                    "provenance_tracker": {
                        "enabled": True,
                        "max_qubits": 20,
                        "hash_precision": 256
                    }
                }
            }
            
            config_file = self.config_dir / "quantum_config.json"
            with open(config_file, 'w') as f:
                import json
                json.dump(quantum_config, f, indent=2)
            
            logger.info(f"✓ Created quantum configuration: {config_file}")
            
            # Environment template
            env_template = """# Quantum LIMIT-Graph v2.0 Environment Variables

# Quantum Computing Backend
QUANTUM_BACKEND=qiskit_aer

# Optional: IBM Quantum Access
# IBMQ_TOKEN=your_ibm_quantum_token_here

# Optional: Google Quantum AI
# GOOGLE_QUANTUM_PROJECT=your_google_project_id

# Optional: Rigetti Quantum Computing  
# RIGETTI_API_KEY=your_rigetti_api_key

# Optional: Amazon Braket
# AWS_ACCESS_KEY_ID=your_aws_access_key
# AWS_SECRET_ACCESS_KEY=your_aws_secret_key
# AWS_DEFAULT_REGION=us-east-1

# Logging Level
LOG_LEVEL=INFO

# Session Configuration
MAX_QUBITS=24
DEFAULT_LANGUAGES=indonesian,arabic,spanish,english
"""
            
            env_file = self.config_dir / ".env.template"
            with open(env_file, 'w') as f:
                f.write(env_template)
            
            logger.info(f"✓ Created environment template: {env_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating config files: {e}")
            return False
    
    def run_quantum_tests(self) -> bool:
        """Run basic quantum functionality tests."""
        logger.info("Running quantum functionality tests...")
        
        try:
            # Import quantum components
            from quantum_integration import QuantumLimitGraph
            
            # Initialize with minimal configuration
            quantum_agent = QuantumLimitGraph(
                languages=['english', 'spanish'],
                max_qubits=8,  # Small for testing
                enable_quantum_walks=True,
                enable_quantum_rlhf=False,  # Skip for quick test
                enable_quantum_context=True,
                enable_quantum_benchmarking=False,  # Skip for quick test
                enable_quantum_provenance=True
            )
            
            # Test basic quantum research
            test_query = "quantum semantic processing test"
            results = quantum_agent.quantum_research(
                test_query, 
                languages=['english'],
                research_depth='quick'
            )
            
            if results and 'synthesis' in results:
                logger.info("✓ Basic quantum research functionality working")
            else:
                logger.error("Quantum research test failed")
                return False
            
            # Test system status
            status = quantum_agent.get_quantum_system_status()
            if status and 'session_id' in status:
                logger.info("✓ Quantum system status working")
            else:
                logger.error("Quantum system status test failed")
                return False
            
            logger.info("✅ All quantum functionality tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Quantum functionality tests failed: {e}")
            return False
    
    def setup_complete(self) -> bool:
        """Complete setup process."""
        logger.info("🚀 Starting Quantum LIMIT-Graph v2.0 Setup...")
        
        # Step 1: Check Python version
        if not self.check_python_version():
            return False
        
        # Step 2: Install requirements
        if not self.install_requirements():
            return False
        
        # Step 3: Verify quantum installations
        if not self.verify_quantum_installation():
            return False
        
        # Step 4: Create configuration files
        if not self.create_config_files():
            return False
        
        # Step 5: Run basic tests
        if not self.run_quantum_tests():
            return False
        
        logger.info("✅ Quantum LIMIT-Graph v2.0 setup completed successfully!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Review configuration files in ./config/")
        logger.info("2. Set up environment variables (copy .env.template to .env)")
        logger.info("3. Run: python -c 'from quantum_integration import QuantumLimitGraph; print(\"Ready!\")'")
        logger.info("4. See README.md for usage examples")
        
        return True

def main():
    """Main setup function."""
    setup = QuantumSetup()
    success = setup.setup_complete()
    
    if not success:
        logger.error("❌ Setup failed. Please check the errors above.")
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()
#!/bin/bash

# Test script for DPLR with verlet/split integrator
# This script runs both reference and test simulations and compares the results

set -e # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
	echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
	echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
	echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
	echo -e "${BLUE}[STEP]${NC} $1"
}

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_status "Starting DPLR verlet/split test in directory: $SCRIPT_DIR"

# Check if required files exist
print_step "Checking required files..."
required_files=("ref/input.lmp" "test/input.lmp" "graph.pb" "system.data" "_test_consistency.py")
for file in "${required_files[@]}"; do
	if [ ! -f "$file" ]; then
		print_error "Required file not found: $file"
		exit 1
	fi
done

# Check if LAMMPS is available
if ! command -v lmp_mpi &>/dev/null; then
	print_error "LAMMPS (lmp_mpi) not found in PATH"
	exit 1
fi

# Check if mpirun is available
if ! command -v mpirun &>/dev/null; then
	print_error "mpirun not found in PATH"
	exit 1
fi

# Check if Python is available
if ! command -v python &>/dev/null; then
	print_error "Python not found in PATH"
	exit 1
fi

# Check if required Python packages are available
print_step "Checking Python dependencies..."
python3 -c "import numpy; import ase" 2>/dev/null || {
	print_error "Required Python packages (numpy, ase) not found"
	exit 1
}

# Clean up previous results
print_step "Cleaning up previous results..."
rm -f ./*/dump.lammpstrj ./*/log.lammps* ./*/screen.* 2>/dev/null || true

# Run reference simulation
print_step "Running reference DPLR simulation (standard DPLR)..."
cd ref
if ! bash run.sh; then
	print_error "Reference simulation failed"
	cd ..
	exit 1
fi
cd ..

# Check if reference output files were created
if [ ! -f "ref/dump.lammpstrj" ]; then
	print_error "Reference dump file not created"
	exit 1
fi

if [ ! -f "ref/log.lammps" ]; then
	print_error "Reference log file not created"
	exit 1
fi

# Run test simulation with verlet/split integrator
print_step "Running test DPLR simulation with verlet/split integrator..."
cd test
if ! bash run.sh; then
	print_error "Test simulation failed"
	cd ..
	exit 1
fi
cd ..

# Check if test output files were created
if [ ! -f "test/dump.lammpstrj" ]; then
	print_error "Test dump file not created"
	exit 1
fi

if [ ! -f "test/log.lammps.0" ]; then
	print_error "Test log file not created"
	exit 1
fi

# Compare results using the consistency test
print_step "Comparing results between reference and test simulations..."
if python _test_consistency.py; then
	print_status "Test PASSED! Results are consistent between reference and test simulations."
else
	print_error "Test FAILED! Results differ between reference and test simulations."
	exit 1
fi

exit 0

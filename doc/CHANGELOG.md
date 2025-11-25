# ec-MLP CHANGELOG

## v1.0.1 (2025-11-25)

### Recent Changes

#### Documentation Updates

- **007432c** - Update documentation with new deployment guide and GitHub workflows
  - Added CI/CD workflows for automated testing and documentation deployment
  - Added comprehensive deployment guide (DEPLOYMENT_GUIDE.md)
  - Updated documentation structure and improved formatting

#### Code Formatting and Cleanup

- **8ffb0dd** - Code formatting improvements
  - Updated formatting in LAMMPS plugin files
  - Consistent code style across C++ files

#### Testing Infrastructure

- **a1496d8** - Added batch test script for LAMMPS

  - New `run_all_tests.sh` script for automated testing
  - Improved test organization with proper gitignore files
- **45f48f1** - Cleaned up test structure for verlet split dplr

  - Reorganized test files and directories
  - Removed redundant test output files
  - Improved test consistency checking
- **589bb99** - Additional formatting improvements

  - Updated pre-commit configuration
  - Code style consistency across multiple files

### Major Feature Developments

#### LAMMPS Integration Improvements

- **7514da1** - Major reconstruction of verlet/split for dplr

  - Complete refactoring of verlet split implementation
  - New modular architecture with separate kspace handling
  - Improved code organization and maintainability
- **c90a21e** - Extended verlet/split/dplr compatibility

  - Made verlet/split/dplr applicable to more kspace styles
  - Enhanced flexibility for different simulation configurations
  - Improved installation and build process

#### Bug Fixes and Enhancements

- **64a7f74** - Added support for pppm/dplr with verlet/split/dplr

  - Corrected self-energy computation for non-pppm/dplr k-space styles
  - Enhanced compatibility with various kspace methods
- **986c6a6** - Fixed bugs in LAMMPS plugin

  - Resolved issues in plugin initialization and execution
  - Improved stability and reliability

#### Examples and Testing

- **c665e6a** - Added comprehensive tests for verlet_split_dplr

  - New bulk water test case with reference data
  - Automated consistency checking between reference and test outputs
  - Complete test infrastructure with input files and expected results
- **5541f3c** - Updated electrode example

  - Added complete electrode simulation example
  - Included system data and configuration files

#### Code Organization

- **3caa204** - Reorganized project files
  - Moved example files to appropriate directories
  - Improved project structure and organization
  - Enhanced maintainability

#### Documentation

- **35e4383** - Updated README and documentation
  - Improved project documentation
  - Enhanced user guides and API documentation

#### Dependency Management

- **f69c5ee** - Added version requirement for numpy
  - Specified numpy<2.0 requirement for compatibility
  - Updated pyproject.toml with dependency constraints

### Project Configuration Updates

#### Build System

- **1ae9213** - Fixed format bug in LAMMPS version number in CMakeLists.txt
- **6d47527** - Format improvements in CMakeLists.txt

#### Project Metadata

- **b4e5629** - Updated README and pyproject.toml
  - Enhanced project description
  - Updated version information and dependencies

### Code Quality Improvements

#### Cleanup and Optimization

- **c99eb9b** - Removed commented-out code

  - Cleaned up legacy code in verlet_split_dplr.cpp
  - Improved code readability
- **057e2bc** - Removed redundant code in verlet/split/dplr

  - Streamlined implementation
  - Reduced code complexity

#### Testing Examples

- **2c8b6ea** - Added comprehensive examples for verlet/split

  - New reference and test examples
  - Complete input/output data for validation
- **2f91847** - Updated existing examples

  - Improved example configurations
  - Enhanced test coverage
- **fd2c20f** - Minor example updates

  - Refined input parameters
  - Improved output formatting

### Technical Details

#### Commit Statistics

- **Total Commits**: 27
- **Time Span**: 6 months ago to 2025-11-25
- **Primary Contributors**: Jia-Xin Zhu, hans
- **Files Changed**: 100+ files across the project
- **Lines Added**: ~150,000+ (including test data)
- **Lines Removed**: ~120,000+ (mostly redundant test outputs)

#### Most Frequently Modified Files

1. `src/lmp/verlet_split_dplr.cpp` - Modified in 8+ commits
2. `src/lmp/ecmlpplugin.cpp` - Modified in 5+ commits
3. `README.md` - Modified in 4+ commits
4. `.pre-commit-config.yaml` - Modified in 3+ commits
5. `src/lmp/CMakeLists.txt` - Modified in 3+ commits

#### New Files Added

- `.github/workflows/ci.yml`
- `.github/workflows/deploy-docs-ghpages.yml`
- `.github/workflows/deploy-docs.yml`
- `DEPLOYMENT_GUIDE.md`
- `tests/lmp/run_all_tests.sh`
- `src/lmp/verlet_split_kspace.cpp`
- `src/lmp/verlet_split_kspace.h`

#### Files Removed

- `src/lmp/mylmpplugin.cpp`
- Various test output files (dump.lammpstrj, log.lammps, etc.)

### Summary

The v1.0.1 release includes significant improvements to the ec-MLP project since v1.0.0, including:

1. **Major Architecture Refactoring**: Complete reconstruction of the verlet/split implementation for better modularity and maintainability.
2. **Enhanced LAMMPS Integration**: Improved compatibility with various kspace styles and better plugin stability.
3. **Comprehensive Testing**: Added extensive test infrastructure with automated validation and consistency checking.
4. **Documentation Improvements**: Enhanced documentation with deployment guides and automated workflows.
5. **Code Quality**: Significant cleanup, formatting improvements, and removal of redundant code.
6. **Example Enhancements**: Added new examples and improved existing ones for better user guidance.

These changes collectively improve the reliability, usability, and maintainability of the ec-MLP project while maintaining backward compatibility with existing workflows.

### Migration Notes

#### Breaking Changes

- None explicitly identified, but users should verify their LAMMPS plugin configurations

#### Compatibility

- Maintained backward compatibility with existing workflows
- Enhanced compatibility with more kspace styles

#### Recommended Actions

1. Update LAMMPS plugin installations
2. Review new documentation for deployment procedures
3. Utilize new automated testing infrastructure for validation

---

## v1.0.0 (2025-11-18)

### Bug Fixes
- **644534d** - Fixed bug for module import

### Features from v1.0.0a1
- Basic ec-MLP functionality
- LAMMPS plugin integration
- TensorFlow modifier support
- PyTorch modifier support
- Basic documentation

---

## v1.0.0a1 (2025-05-08)

### Initial Features
- **78e52dc** - Initial project setup
- **9514a80** - Added DipoleChargeBetaModifier for GPU-accelerated dipole_charge modifier
- **7fc0e7c** - Added multiple backend support for dipole_charge modifier
- **01efc7f** - Updated README with project information
- **cfba236** - Added examples to README
- **992445d** - Changed source from GitHub to Gitee
- **8f7c4c9** - Updated installation source
- **2749989** - Added sync workflow
- **509f04c** - Added PyTorch dipole_charge data modifier

### Core Functionality
- Dipole charge modifiers for both TensorFlow and PyTorch backends
- LAMMPS integration for molecular dynamics simulations
- GPU acceleration support
- Basic documentation and examples

---

*This changelog will be updated as new releases are made. For the most recent changes, please refer to the git commit history.*

# cn8k-previs

Simulation system for developing and testing interactive content that allows visualization of the user experience before its implementation in the exhibition.

## Main Features

- Navigation in virtual exhibition room
- Tracking system simulation with TUIO data output
- Video input for two projection surfaces
- Support for interactive content development and testing
- Spatialized audio simulation
- Video and screenshot export capabilities

## System Requirements

### Operating System
- Windows 10/11
- macOS 10.15 or later

### Hardware
- Vulkan 1.1 compatible GPU
- Minimum 4GB GPU memory
- Updated drivers

### Supported GPUs

**NVIDIA:**
- GeForce 800 Series or higher
- Quadro K Series or higher
- Driver 472.47 or newer

**AMD:**
- HD 7000 Series or higher
- W or V Series or higher

**Intel:**
- Intel 500 GPUs and newer

**Apple:**
- Mac Pro / iMac / Mac Mini / MacBook Pro / MacBook Air 2015+
- Compatible with macOS 12+

## Installation

1. Install TouchDesigner v2023.12000 or later
2. For non-commercial license:
   - Username: `cn8k-previs`
   - Password: `KRR!85b^knjJ"M]`
3. Download the previewer package
4. Extract and run `previs.toe`

## Usage

### Navigation
- Left click + drag: Adjust observer view
- Right click + drag: Adjust observer position
- WASD keys: Room movement
- Keys 1-2: Predefined views

### Content Loading
Supports multiple input types:
- Video (local files)
- NDI (network streaming)
- Spout (local streaming)
- Screen capture
- Notch Blocks
- Spatialized audio

### Tracking Simulation
- LiDAR sensors simulation
- TUIO v1.1 data output
- Diagnostic visualization
- Visitor presence control

### Export
- Screenshots (PNG, JPG)
- First-person view video
- 360° video (requires metadata injection)

## Configuration

### Save / Load
- Save configurations in JSON format
- Load previously saved configurations
- Reset to default values

### Tracking
- UDP port 3333 and 3334 for TUIO
- 4 LiDAR sensors simulation
- Configurable OSC simulation data output

## Limitations

- Non-commercial version limits resolution to 720p
- Spatialized audio with reverb only available on Windows

## Support

For more information and updates, refer to the complete documentation.
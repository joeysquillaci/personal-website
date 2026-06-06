# ESP32 Air Pointer

A Wii Remote–style Bluetooth air mouse built using an ESP32 and an Adafruit BNO055 9-axis IMU.

The device tracks controller orientation in space and presents itself to a computer as a Bluetooth HID mouse. Controller yaw and pitch are mapped to a virtual cursor position, allowing the user to point at the screen by physically aiming the controller.

Additional directional buttons provide manual cursor panning for fine adjustment and navigation.

---

## Features

- Bluetooth HID mouse using ESP32 BLE
- BNO055 sensor fusion for absolute orientation tracking
- Wii Remote–style pointing behavior
- Yaw controls horizontal cursor movement
- Pitch controls vertical cursor movement
- Deadzone filtering to reduce jitter
- Low-pass filtering for smoother movement
- Automatic rejection of large orientation glitches
- Manual recentering via serial command
- Directional button panning
- Virtual cursor tracking to approximate absolute positioning
- Artificial software cursor bounds removed to prevent early edge clipping

---

## Hardware

### Main Components

- ESP32 development board
- Adafruit BNO055 9-axis IMU
- 3.7V Li-Ion battery
- Li-Ion charging module
- Buck/boost converter
- Four directional buttons

### BNO055 Wiring

| BNO055 | ESP32 |
|----------|----------|
| VIN | 3.3V |
| GND | GND |
| SDA | GPIO21 |
| SCL | GPIO22 |

I²C configuration:

```cpp
Wire.begin(21, 22);
Wire.setClock(400000);
```

BNO055 I²C address:

```cpp
0x28
```

### Directional Buttons

| Function | ESP32 Pin |
|-----------|-----------|
| Up | GPIO13 |
| Down | GPIO14 |
| Left | GPIO27 |
| Right | GPIO26 |

Buttons are configured using INPUT_PULLUP and connect to GND when pressed.

---

## Bluetooth

Device Name:

```text
ESP32 Air Pointer
```

Manufacturer:

```text
Joey
```

The device appears as a standard Bluetooth HID mouse.

---

## Cursor Mapping

The controller operates as a virtual absolute pointer:

1. Orientation is read from the BNO055.
2. Relative yaw and pitch are calculated.
3. Angles are converted into a target screen position.
4. The cursor continuously moves toward that target.

This produces behavior similar to a Wii Remote.

### Horizontal Mapping

```cpp
yawEdgeDeg = 60.0;
```

### Vertical Mapping

```cpp
pitchEdgeDeg = 33.0;
```

Vertical movement is intentionally more sensitive than horizontal movement.

---

## Filtering

### Deadzone

```cpp
yawDeadzoneDeg = 0.4;
pitchDeadzoneDeg = 0.4;
```

### Low Pass Filter

```cpp
angleSmoothing = 0.18;
```

Lower values increase smoothness. Higher values increase responsiveness.

### Yaw Jump Rejection

Large orientation discontinuities are ignored.

Example:

```text
20° -> 130°
```

Threshold:

```cpp
maxYawJumpDeg = 35.0;
```

---

## Cursor Tuning

```cpp
followGain = 0.11;
maxMoveStep = 18;
virtualMovementScale = 0.45;
```

These values control responsiveness and compensate for operating-system mouse acceleration.

---

## Button Panning

Directional buttons continuously shift the target position while held.

```cpp
buttonPanStep = 4.0;
```

This allows fine adjustment without changing controller orientation.

---

## Recentering

To recenter:

1. Aim the controller where you want screen center.
2. Open Serial Monitor.
3. Send:

```text
c
```

The software will:
- Reset yaw and pitch references
- Reset button offsets
- Move the cursor back to screen center

---

## Known Limitations

### Virtual Cursor Tracking

The ESP32 cannot read the actual host cursor position.

Instead it maintains an internal estimate using:

```cpp
virtualX
virtualY
```

Because of this:

- Minor drift can accumulate
- Mouse acceleration affects accuracy
- Periodic recentering may be required

### BNO055 Yaw Stability

The BNO055 occasionally produces sudden yaw discontinuities. Filtering reduces their impact but cannot eliminate them completely.

---

## Future Improvements

- Left/right click buttons
- Scroll wheel emulation
- Battery monitoring
- Automatic drift compensation
- Host-side cursor synchronization
- IR beacon tracking similar to the original Wii Remote
- Custom BLE HID implementation

---

## Tuned Parameters

```cpp
yawEdgeDeg            = 60.0
pitchEdgeDeg          = 33.0

yawDeadzoneDeg        = 0.4
pitchDeadzoneDeg      = 0.4

maxYawJumpDeg         = 35.0
maxPitchJumpDeg       = 35.0

angleSmoothing        = 0.18

followGain            = 0.11
maxMoveStep           = 18

virtualMovementScale  = 0.45

buttonPanStep         = 4.0

updateMs              = 5
```

These values were tuned on a 3440×1440 ultrawide display running macOS.

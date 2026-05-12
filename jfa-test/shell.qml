import QtQuick
import Quickshell
import Quickshell.Wayland

ShellRoot {
    id: root
    property real cx: 0
    property real cy: 0
    // Scroll-wheel adjustable scale factor applied uniformly to shape sizes
    // and inter-shape offsets so the test grid scales as a whole.
    property real shapeScale: 1.0

    PanelWindow {
        id: window
        WlrLayershell.namespace: "quickshell"
        WlrLayershell.layer: WlrLayershell.Top
        anchors {
            top: true
            bottom: true
            left: true
            right: true
        }
        color: "transparent"
        exclusionMode: ExclusionMode.Ignore

        MouseArea {
            anchors.fill: parent
            hoverEnabled: true
            onPositionChanged: function (mouse) {
                root.cx = mouse.x
                root.cy = mouse.y
            }
            onWheel: function (wheel) {
                // 120 units per detent on a standard wheel. ×1.1 per detent
                // gives ~10% growth per notch; clamp keeps things sane.
                var step = Math.pow(1.1, wheel.angleDelta.y / 120)
                root.shapeScale = Math.max(0.1, Math.min(5.0, root.shapeScale * step))
                wheel.accepted = true
            }
        }

        BackgroundEffect.blurRegion: Region {
            // Solid square next to the mouse cursor.
            Region {
                x: root.cx - 200 * root.shapeScale
                y: root.cy - 200 * root.shapeScale
                width: 400 * root.shapeScale
                height: 400 * root.shapeScale
            }

            // Hollow square (square-with-square-hole) to the right of the
            // solid one. Built from four rectangles forming a frame:
            //   outer extent: 400×400, offset +500 px to the right of cx
            //   inner hole:   200×200, centered
            //
            //   ┌─────────────┐  ── top strip (400 × 100)
            //   │             │
            //   ├──┐       ┌──┤  ── left strip (100 × 200), right strip (100 × 200)
            //   │  │       │  │
            //   ├──┘       └──┤
            //   │             │
            //   └─────────────┘  ── bottom strip (400 × 100)
            Region {
                x: root.cx + 300 * root.shapeScale
                y: root.cy - 200 * root.shapeScale
                width: 400 * root.shapeScale
                height: 100 * root.shapeScale
            }
            Region {
                x: root.cx + 300 * root.shapeScale
                y: root.cy + 100 * root.shapeScale
                width: 400 * root.shapeScale
                height: 100 * root.shapeScale
            }
            Region {
                x: root.cx + 300 * root.shapeScale
                y: root.cy - 100 * root.shapeScale
                width: 100 * root.shapeScale
                height: 200 * root.shapeScale
            }
            Region {
                x: root.cx + 600 * root.shapeScale
                y: root.cy - 100 * root.shapeScale
                width: 100 * root.shapeScale
                height: 200 * root.shapeScale
            }

            // Cross (two overlapping rectangles) below the solid square.
            // Center at (cx, cy + 500). 400 × 400 overall extent with
            // 100 px arm thickness.
            Region {
                x: root.cx - 200 * root.shapeScale
                y: root.cy + 450 * root.shapeScale
                width: 400 * root.shapeScale
                height: 100 * root.shapeScale
            }
            Region {
                x: root.cx - 50 * root.shapeScale
                y: root.cy + 300 * root.shapeScale
                width: 100 * root.shapeScale
                height: 400 * root.shapeScale
            }

            // Square with four holes (bottom-right of the grid). Built by
            // combining a hollow frame with a cross inside its hole: the
            // cross's arms connect to the inner edges of the frame on all
            // four sides, splitting the central hole into four isolated
            // corner pockets — i.e. a 400 × 400 solid square punctured by
            // four 75 × 75 holes.
            //
            // Outer frame: 400 × 400 outer, 200 × 200 inner hole,
            // 100 px-thick walls, centered at (cx + 500, cy + 500).
            Region {
                x: root.cx + 300 * root.shapeScale
                y: root.cy + 300 * root.shapeScale
                width: 400 * root.shapeScale
                height: 100 * root.shapeScale
            }
            Region {
                x: root.cx + 300 * root.shapeScale
                y: root.cy + 600 * root.shapeScale
                width: 400 * root.shapeScale
                height: 100 * root.shapeScale
            }
            Region {
                x: root.cx + 300 * root.shapeScale
                y: root.cy + 400 * root.shapeScale
                width: 100 * root.shapeScale
                height: 200 * root.shapeScale
            }
            Region {
                x: root.cx + 600 * root.shapeScale
                y: root.cy + 400 * root.shapeScale
                width: 100 * root.shapeScale
                height: 200 * root.shapeScale
            }
            // Cross inside the hole: 50 px arm thickness, spans the full
            // 200 × 200 inner extent so its tips meet the frame.
            Region {
                x: root.cx + 400 * root.shapeScale
                y: root.cy + 475 * root.shapeScale
                width: 200 * root.shapeScale
                height: 50 * root.shapeScale
            }
            Region {
                x: root.cx + 475 * root.shapeScale
                y: root.cy + 400 * root.shapeScale
                width: 50 * root.shapeScale
                height: 200 * root.shapeScale
            }
        }
    }

    Item {
        focus: true
        Keys.onEscapePressed: Qt.quit()
    }
}

import QtQuick
import Quickshell
import Quickshell.Wayland

ShellRoot {
    id: root
    property real cx: 0
    property real cy: 0

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
        }

        BackgroundEffect.blurRegion: Region {
            // Solid square next to the mouse cursor.
            Region {
                x: root.cx - 200
                y: root.cy - 200
                width: 400
                height: 400
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
                x: root.cx + 300
                y: root.cy - 200
                width: 400
                height: 100
            }
            Region {
                x: root.cx + 300
                y: root.cy + 100
                width: 400
                height: 100
            }
            Region {
                x: root.cx + 300
                y: root.cy - 100
                width: 100
                height: 200
            }
            Region {
                x: root.cx + 600
                y: root.cy - 100
                width: 100
                height: 200
            }

            // Cross (two overlapping rectangles) below the solid square.
            // Center at (cx, cy + 500). 400 × 400 overall extent with
            // 100 px arm thickness.
            Region {
                x: root.cx - 200
                y: root.cy + 450
                width: 400
                height: 100
            }
            Region {
                x: root.cx - 50
                y: root.cy + 300
                width: 100
                height: 400
            }
        }
    }

    Item {
        focus: true
        Keys.onEscapePressed: Qt.quit()
    }
}

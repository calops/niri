import QtQuick
import Quickshell
import Quickshell.Wayland

ShellRoot {
    id: root

    property real cx: 640
    property real cy: 700
    property real shapeScale: 1.0

    DonkeyRegion {
        id: donkeyRegion
        centerX: root.cx
        centerY: root.cy
        shapeScale: root.shapeScale
    }

    PanelWindow {
        WlrLayershell.namespace: "donkey-test"
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
            onPositionChanged: function(mouse) {
                root.cx = Math.round(mouse.x)
                root.cy = Math.round(mouse.y)
            }
            onWheel: function(wheel) {
                var step = Math.pow(1.1, wheel.angleDelta.y / 120)
                root.shapeScale = Math.max(0.25, Math.min(2.5, root.shapeScale * step))
                wheel.accepted = true
            }
        }

        BackgroundEffect.blurRegion: donkeyRegion
    }

    Item {
        focus: true
        Keys.onEscapePressed: Qt.quit()
    }
}

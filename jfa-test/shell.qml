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
            Region {
                x: root.cx - 200
                y: root.cy - 200
                width: 400
                height: 400
            }
        }
    }

    Item {
        focus: true
        Keys.onEscapePressed: Qt.quit()
    }
}

import QtQuick
import Quickshell
import Quickshell.Wayland

ShellRoot {
    id: root
    property real cx: 0
    property real cy: 0
    property real shapeScale: 1.0
    property real overlap: 1.0

    readonly property int circBaseR: 200
    readonly property int columnWidth: 2
    readonly property int numColumns: Math.floor(2 * circBaseR / columnWidth)

    property var circleRegion: null

    function rebuildCircleRegion() {
        if (circleRegion !== null) {
            circleRegion.destroy()
            circleRegion = null
        }

        var r = circBaseR
        var src = "import Quickshell; Region {\n"
        for (var i = 0; i < numColumns; i++) {
            var colCenterX = -r + columnWidth / 2.0 + i * columnWidth
            var t = Math.abs(colCenterX) / r
            if (t > 1.0) continue
            var halfChord = Math.sqrt(1.0 - t * t) * r
            var colTop = -halfChord
            var colBottom = halfChord

            var rx = Math.round(cx + (colCenterX - columnWidth / 2.0) * shapeScale - overlap)
            var ry = Math.round(cy + colTop * shapeScale - overlap)
            var rw = Math.round(columnWidth * shapeScale + 2 * overlap)
            var rh = Math.round((colBottom - colTop) * shapeScale + 2 * overlap)

            src += "    Region { x: " + rx + "; y: " + ry
                + "; width: " + rw + "; height: " + rh + " }\n"
        }
        src += "}\n"

        circleRegion = Qt.createQmlObject(src, root, "CircleRegion")
    }

    onCxChanged: rebuildCircleRegion()
    onCyChanged: rebuildCircleRegion()
    onShapeScaleChanged: rebuildCircleRegion()
    Component.onCompleted: rebuildCircleRegion()

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
                root.cx = Math.round(mouse.x)
                root.cy = Math.round(mouse.y)
            }
            onWheel: function (wheel) {
                var step = Math.pow(1.1, wheel.angleDelta.y / 120)
                root.shapeScale = Math.max(0.1, Math.min(5.0, root.shapeScale * step))
                wheel.accepted = true
            }
        }

        BackgroundEffect.blurRegion: root.circleRegion
    }

    Item {
        focus: true
        Keys.onEscapePressed: Qt.quit()
    }
}

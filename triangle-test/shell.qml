import QtQuick
import Quickshell
import Quickshell.Wayland

ShellRoot {
    id: root
    property real cx: 0
    property real cy: 0
    property real shapeScale: 1.0
    property real overlap: 1.0

    readonly property int triBaseW: 400
    readonly property int triBaseH: 350
    readonly property int columnWidth: 2
    readonly property int numColumns: Math.floor(triBaseW / columnWidth)

    // The parent Region containing all triangle columns, assembled by
    // building a QML source string and feeding it to Qt.createQmlObject.
    // Dirty, but bypasses the static-children-only limitation entirely.
    property var triangleRegion: null

    function rebuildTriangleRegion() {
        if (triangleRegion !== null) {
            triangleRegion.destroy()
            triangleRegion = null
        }

        var halfW = triBaseW / 2.0
        var halfH = triBaseH / 2.0
        var src = "import Quickshell; Region {\n"
        for (var i = 0; i < numColumns; i++) {
            var colCenterX = -halfW + columnWidth / 2.0 + i * columnWidth
            var t = Math.abs(colCenterX) / halfW
            var colTop = -halfH + t * triBaseH
            var colBottom = halfH

            var rx = Math.round(cx + (colCenterX - columnWidth / 2.0) * shapeScale - overlap)
            var ry = Math.round(cy + colTop * shapeScale - overlap)
            var rw = Math.round(columnWidth * shapeScale + 2 * overlap)
            var rh = Math.round((colBottom - colTop) * shapeScale + 2 * overlap)

            src += "    Region { x: " + rx + "; y: " + ry
                + "; width: " + rw + "; height: " + rh + " }\n"
        }
        src += "}\n"

        triangleRegion = Qt.createQmlObject(src, root, "TriangleRegion")
    }

    onCxChanged: rebuildTriangleRegion()
    onCyChanged: rebuildTriangleRegion()
    onShapeScaleChanged: rebuildTriangleRegion()
    Component.onCompleted: rebuildTriangleRegion()

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

        BackgroundEffect.blurRegion: root.triangleRegion
    }

    Item {
        focus: true
        Keys.onEscapePressed: Qt.quit()
    }
}

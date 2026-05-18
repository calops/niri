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

    readonly property var circleColumns: {
        var cols = [];
        var r = circBaseR;
        for (var i = 0; i < numColumns; i++) {
            var colCenterX = -r + columnWidth / 2.0 + i * columnWidth;
            var t = Math.abs(colCenterX) / r;
            if (t > 1.0)
                continue;
            var halfChord = Math.sqrt(1.0 - t * t) * r;
            cols.push({
                rx: colCenterX - columnWidth / 2.0,
                ry: -halfChord,
                rw: columnWidth,
                rh: 2 * halfChord
            });
        }
        cols;
    }

    property var circleItems: []

    Component {
        id: columnComp
        Item {
            opacity: 0
            x: Math.round(col.rx * root.shapeScale - root.overlap)
            y: Math.round(col.ry * root.shapeScale - root.overlap)
            width: Math.round(col.rw * root.shapeScale + 2 * root.overlap)
            height: Math.round(col.rh * root.shapeScale + 2 * root.overlap)
            property var col
        }
    }

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
                root.cx = Math.round(mouse.x);
                root.cy = Math.round(mouse.y);
            }
            onWheel: function (wheel) {
                var step = Math.pow(1.1, wheel.angleDelta.y / 120);
                root.shapeScale = Math.max(0.1, Math.min(5.0, root.shapeScale * step));
                wheel.accepted = true;
            }
        }

        Item {
            id: circleContainer
            x: root.cx
            y: root.cy

            Component.onCompleted: {
                var items = [];
                for (var i = 0; i < root.circleColumns.length; i++) {
                    var item = columnComp.createObject(this, {
                        col: root.circleColumns[i]
                    });
                    items.push(item);
                }
                root.circleItems = items;
            }
        }

        BackgroundEffect.blurRegion: Region {
            items: root.circleItems
        }
    }

    Item {
        focus: true
        Keys.onEscapePressed: Qt.quit()
    }
}

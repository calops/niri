import QtQuick
import Quickshell
import Quickshell.Wayland

ShellRoot {
    id: root
    property real cx: 500
    property real cy: 500

    readonly property int lensRadius: 180
    readonly property int rimThickness: 14
    readonly property int handleWidth: 26
    readonly property int handleLength: 200
    readonly property int maxExtent: lensRadius + rimThickness + handleLength + handleWidth

    PanelWindow {
        id: window
        WlrLayershell.namespace: "magnify-glass"
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
        }

        BackgroundEffect.blurRegion: Region {
            x: Math.round(root.cx)
            y: Math.round(root.cy)
            Region { x: -180; y: -38; width: 8; height: 75 }
            Region { x: -172; y: -65; width: 8; height: 129 }
            Region { x: -164; y: -82; width: 8; height: 165 }
            Region { x: -156; y: -96; width: 8; height: 193 }
            Region { x: -148; y: -108; width: 8; height: 216 }
            Region { x: -140; y: -118; width: 8; height: 236 }
            Region { x: -132; y: -127; width: 8; height: 253 }
            Region { x: -124; y: -134; width: 8; height: 268 }
            Region { x: -116; y: -141; width: 8; height: 282 }
            Region { x: -108; y: -147; width: 8; height: 294 }
            Region { x: -100; y: -152; width: 8; height: 305 }
            Region { x: -92; y: -157; width: 8; height: 314 }
            Region { x: -84; y: -161; width: 8; height: 322 }
            Region { x: -76; y: -165; width: 8; height: 330 }
            Region { x: -68; y: -168; width: 8; height: 336 }
            Region { x: -60; y: -171; width: 8; height: 342 }
            Region { x: -52; y: -173; width: 8; height: 347 }
            Region { x: -44; y: -175; width: 8; height: 351 }
            Region { x: -36; y: -177; width: 8; height: 354 }
            Region { x: -28; y: -178; width: 8; height: 357 }
            Region { x: -20; y: -179; width: 8; height: 359 }
            Region { x: -12; y: -180; width: 8; height: 360 }
            Region { x: -4; y: -180; width: 8; height: 360 }
            Region { x: 4; y: -180; width: 8; height: 360 }
            Region { x: 12; y: -179; width: 8; height: 359 }
            Region { x: 20; y: -178; width: 8; height: 357 }
            Region { x: 28; y: -177; width: 8; height: 354 }
            Region { x: 36; y: -175; width: 8; height: 351 }
            Region { x: 44; y: -173; width: 8; height: 347 }
            Region { x: 52; y: -171; width: 8; height: 342 }
            Region { x: 60; y: -168; width: 8; height: 336 }
            Region { x: 68; y: -165; width: 8; height: 330 }
            Region { x: 76; y: -161; width: 8; height: 322 }
            Region { x: 84; y: -157; width: 8; height: 314 }
            Region { x: 92; y: -152; width: 8; height: 305 }
            Region { x: 100; y: -147; width: 8; height: 294 }
            Region { x: 108; y: -141; width: 8; height: 282 }
            Region { x: 116; y: -134; width: 8; height: 268 }
            Region { x: 124; y: -127; width: 8; height: 253 }
            Region { x: 132; y: -118; width: 8; height: 236 }
            Region { x: 140; y: -108; width: 8; height: 216 }
            Region { x: 148; y: -96; width: 8; height: 193 }
            Region { x: 156; y: -82; width: 8; height: 165 }
            Region { x: 164; y: -65; width: 8; height: 129 }
            Region { x: 172; y: -38; width: 8; height: 75 }
        }

        // Tracking item: only moves via x/y, Canvas paints once.
        Item {
            x: Math.round(root.cx - root.maxExtent)
            y: Math.round(root.cy - root.maxExtent)
            width: root.maxExtent * 2
            height: root.maxExtent * 2

            Canvas {
                id: glassCanvas
                anchors.fill: parent
                renderStrategy: Canvas.Cooperative
                antialiasing: true

                Component.onCompleted: requestPaint()

                onPaint: {
                    var ctx = getContext("2d")
                    ctx.clearRect(0, 0, width, height)

                    var hx = width / 2
                    var hy = height / 2
                    var r = root.lensRadius
                    var rt = root.rimThickness
                    var hw = root.handleWidth
                    var hl = root.handleLength

                    var angle = Math.PI / 4
                    var handleStart = r + rt - 2
                    var ferruleLen = 24

                    ctx.save()

                    // RIM
                    ctx.beginPath()
                    ctx.arc(hx, hy, r + rt + 4, 0, 2 * Math.PI)
                    ctx.shadowColor = "rgba(0,0,0,0.5)"
                    ctx.shadowBlur = 28
                    ctx.shadowOffsetX = 2
                    ctx.shadowOffsetY = 4
                    ctx.fillStyle = "rgba(0,0,0,0.2)"
                    ctx.fill()
                    ctx.shadowColor = "transparent"
                    ctx.shadowBlur = 0
                    ctx.shadowOffsetX = 0
                    ctx.shadowOffsetY = 0

                    ctx.beginPath()
                    ctx.arc(hx, hy, r + rt, 0, 2 * Math.PI)
                    var outerRimGrad = ctx.createRadialGradient(
                        hx - r * 0.3, hy - r * 0.3, r * 0.1,
                        hx, hy, r + rt
                    )
                    outerRimGrad.addColorStop(0, "#d4af37")
                    outerRimGrad.addColorStop(0.4, "#b8860b")
                    outerRimGrad.addColorStop(1, "#654321")
                    ctx.fillStyle = outerRimGrad
                    ctx.fill()

                    ctx.beginPath()
                    ctx.arc(hx, hy, r + rt * 0.6, 0, 2 * Math.PI)
                    var innerRimGrad = ctx.createRadialGradient(
                        hx - r * 0.3, hy - r * 0.3, r * 0.1,
                        hx, hy, r + rt * 0.6
                    )
                    innerRimGrad.addColorStop(0, "#ffd700")
                    innerRimGrad.addColorStop(0.5, "#daa520")
                    innerRimGrad.addColorStop(1, "#8b6914")
                    ctx.fillStyle = innerRimGrad
                    ctx.fill()

                    // Punch out lens area
                    ctx.globalCompositeOperation = 'destination-out'
                    ctx.beginPath()
                    ctx.arc(hx, hy, r, 0, 2 * Math.PI)
                    ctx.fill()
                    ctx.globalCompositeOperation = 'source-over'

                    // Bevel rings
                    ctx.beginPath()
                    ctx.arc(hx, hy, r + 2, 0, 2 * Math.PI)
                    ctx.strokeStyle = "#4a3b2a"
                    ctx.lineWidth = 3
                    ctx.stroke()

                    ctx.beginPath()
                    ctx.arc(hx, hy, r, 0, 2 * Math.PI)
                    ctx.strokeStyle = "#2a1d15"
                    ctx.lineWidth = 1
                    ctx.stroke()

                    // FERRULE
                    ctx.save()
                    ctx.translate(hx, hy)
                    ctx.rotate(angle)
                    ctx.lineCap = "round"
                    ctx.lineWidth = hw + 4
                    ctx.strokeStyle = "#7a6a5a"
                    ctx.beginPath()
                    ctx.moveTo(handleStart, 0)
                    ctx.lineTo(handleStart + ferruleLen, 0)
                    ctx.stroke()

                    ctx.lineWidth = hw + 2
                    ctx.strokeStyle = "#b0a090"
                    ctx.beginPath()
                    ctx.moveTo(handleStart, 0)
                    ctx.lineTo(handleStart + ferruleLen, 0)
                    ctx.stroke()
                    ctx.restore()

                    // HANDLE
                    ctx.save()
                    ctx.translate(hx, hy)
                    ctx.rotate(angle)
                    ctx.shadowColor = "rgba(0,0,0,0.4)"
                    ctx.shadowBlur = 18
                    ctx.shadowOffsetX = 3
                    ctx.shadowOffsetY = 3
                    ctx.lineCap = "round"
                    ctx.lineWidth = hw
                    ctx.strokeStyle = "#1a0f08"
                    ctx.beginPath()
                    ctx.moveTo(handleStart, 0)
                    ctx.lineTo(handleStart + hl, 0)
                    ctx.stroke()
                    ctx.shadowColor = "transparent"
                    ctx.shadowBlur = 0
                    ctx.shadowOffsetX = 0
                    ctx.shadowOffsetY = 0
                    ctx.restore()

                    ctx.save()
                    ctx.translate(hx, hy)
                    ctx.rotate(angle)
                    ctx.lineCap = "round"
                    ctx.lineWidth = hw
                    var woodGrad = ctx.createLinearGradient(
                        handleStart, -hw/2,
                        handleStart + hl, hw/2
                    )
                    woodGrad.addColorStop(0, "#3d2b1f")
                    woodGrad.addColorStop(0.35, "#5c4033")
                    woodGrad.addColorStop(0.65, "#4a332a")
                    woodGrad.addColorStop(1, "#2a1d15")
                    ctx.strokeStyle = woodGrad
                    ctx.beginPath()
                    ctx.moveTo(handleStart, 0)
                    ctx.lineTo(handleStart + hl, 0)
                    ctx.stroke()

                    ctx.strokeStyle = "rgba(0,0,0,0.12)"
                    ctx.lineWidth = 1
                    for (var gi = 0; gi < 3; gi++) {
                        var gy = -hw/2 + (gi + 1) * (hw / 4)
                        ctx.beginPath()
                        ctx.moveTo(handleStart + hl * 0.1, gy)
                        ctx.lineTo(handleStart + hl * 0.9, gy + (gi % 2 === 0 ? 2 : -2))
                        ctx.stroke()
                    }

                    ctx.strokeStyle = "rgba(255,255,255,0.07)"
                    ctx.lineWidth = hw / 3
                    ctx.beginPath()
                    ctx.moveTo(handleStart + 2, -hw/4)
                    ctx.lineTo(handleStart + hl - 2, -hw/4)
                    ctx.stroke()
                    ctx.restore()

                    // GLASS
                    ctx.beginPath()
                    ctx.arc(hx, hy, r - 1, 0, 2 * Math.PI)
                    var glassGrad = ctx.createRadialGradient(hx, hy, 0, hx, hy, r)
                    glassGrad.addColorStop(0, "rgba(200,220,255,0.06)")
                    glassGrad.addColorStop(0.7, "rgba(180,200,230,0.10)")
                    glassGrad.addColorStop(1, "rgba(160,180,210,0.14)")
                    ctx.fillStyle = glassGrad
                    ctx.fill()

                    ctx.beginPath()
                    ctx.arc(hx - r * 0.25, hy - r * 0.25, r * 0.75, Math.PI * 1.05, Math.PI * 1.55)
                    ctx.strokeStyle = "rgba(255,255,255,0.30)"
                    ctx.lineWidth = 10
                    ctx.lineCap = "round"
                    ctx.stroke()

                    ctx.beginPath()
                    ctx.arc(hx + r * 0.3, hy + r * 0.25, r * 0.2, Math.PI * 0.15, Math.PI * 0.55)
                    ctx.strokeStyle = "rgba(255,255,255,0.10)"
                    ctx.lineWidth = 4
                    ctx.lineCap = "round"
                    ctx.stroke()

                    ctx.beginPath()
                    ctx.arc(hx - r * 0.35, hy - r * 0.35, 4, 0, 2 * Math.PI)
                    ctx.fillStyle = "rgba(255,255,255,0.25)"
                    ctx.fill()

                    ctx.restore()
                }
            }
        }
    }

    Item {
        focus: true
        Keys.onEscapePressed: Qt.quit()
    }
}

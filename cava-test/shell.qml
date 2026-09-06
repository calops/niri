import QtQuick
import Quickshell
import Quickshell.Io
import Quickshell.Wayland

ShellRoot {
    id: root

    readonly property int cavaBarCount: 48
    property real detectedRefreshRate: 0
    property bool cavaStarted: false
    readonly property real cavaFramerateOverride: Number(Quickshell.env("NIRI_CAVA_FRAMERATE"))
    readonly property bool hasCavaFramerateOverride:
        isFinite(cavaFramerateOverride) && cavaFramerateOverride > 0
    readonly property int targetFramerate: hasCavaFramerateOverride
        ? Math.round(cavaFramerateOverride)
        : detectedRefreshRate > 0
            ? Math.round(detectedRefreshRate)
            : 60
    readonly property int surfaceColumnWidth: 1
    readonly property int surfaceBarCount: Math.max(1, Math.ceil(root.visualizationWidth / root.surfaceColumnWidth))
    readonly property int barOverlap: 1
    readonly property int barPitch: root.surfaceColumnWidth
    readonly property int bottomInset: 0
    readonly property int minimumBarHeight: 0
    readonly property int visualizationHeight: Math.max(
        root.minimumBarHeight,
        Math.floor(window.screen.height * 0.5)
    )
    readonly property int visualizationWidth: window.width
    readonly property int trailingWidth: root.visualizationWidth - root.barPitch * (root.surfaceBarCount - 1)
    readonly property int visualizationX: 0

    property var frequencies: []
    property var barItems: []
    property var pendingFrame: []

    Component {
        id: barComponent

        Item {
            required property int barIndex

            readonly property int visualHeight: root.barHeight(barIndex)

            x: root.visualizationX + barIndex * root.barPitch
            y: window.height - root.bottomInset - visualHeight
            width: barIndex === root.surfaceBarCount - 1 ? root.trailingWidth : root.barPitch + root.barOverlap
            height: visualHeight
            opacity: 0
        }
    }

    function interpolatedFrequency(index) {
        const sourceCount = frequencies.length
        if (sourceCount === 0) return 0
        if (sourceCount === 1 || surfaceBarCount === 1) return frequencies[0]

        const sourcePosition = index * (sourceCount - 1) / (surfaceBarCount - 1)
        const center = Math.floor(sourcePosition)
        const t = sourcePosition - center
        const p0 = frequencies[Math.max(0, center - 1)]
        const p1 = frequencies[center]
        const p2 = frequencies[Math.min(sourceCount - 1, center + 1)]
        const p3 = frequencies[Math.min(sourceCount - 1, center + 2)]
        const t2 = t * t
        const t3 = t2 * t
        const frequency = 0.5 * (
            2 * p1
            + (-p0 + p2) * t
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
            + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
        )
        return Math.max(0, Math.min(1, frequency))
    }

    function barHeight(index) {
        return Math.round(
            minimumBarHeight
            + (visualizationHeight - minimumBarHeight)
            * Math.pow(interpolatedFrequency(index), 0.58)
        )
    }

    function acceptCavaValue(value) {
        pendingFrame.push(value / 100.0)
        if (pendingFrame.length === cavaBarCount) {
            frequencies = pendingFrame
            pendingFrame = []
        }
    }

    function startCava() {
        if (cavaStarted) return

        cavaStarted = true
        const detected = detectedRefreshRate > 0
            ? `${detectedRefreshRate} Hz`
            : "an unknown refresh rate"
        const override = hasCavaFramerateOverride ? " (NIRI_CAVA_FRAMERATE override)" : ""
        console.info(`cava-glass: detected ${detected} on ${window.screen.name}; using ${targetFramerate} FPS${override}`)
        cavaProcess.running = true
    }

    function acceptOutputInfo(text) {
        try {
            const outputs = JSON.parse(text)
            const output = outputs[window.screen.name]
            if (output && output.current_mode >= 0 && output.current_mode < output.modes.length) {
                const refreshRate = Number(output.modes[output.current_mode].refresh_rate)
                if (refreshRate > 0) detectedRefreshRate = refreshRate / 1000
            }
        } catch (error) {
            console.warn(`cava-glass: could not read output refresh rate: ${error}`)
        }
        startCava()
    }

    Component.onCompleted: {
        outputQuery.running = true
        const initialFrame = []
        const items = []
        for (let index = 0; index < cavaBarCount; index++)
            initialFrame.push(0)
        for (let index = 0; index < surfaceBarCount; index++) {
            items.push(barComponent.createObject(barContainer, { barIndex: index }))
        }
        frequencies = initialFrame
        barItems = items
        // Cava starts after the output query returns.
    }

    PanelWindow {
        id: window
        WlrLayershell.namespace: "cava-glass"
        WlrLayershell.layer: WlrLayershell.Top
        anchors {
            bottom: true
            left: true
            right: true
        }
        implicitWidth: screen.width
        implicitHeight: root.visualizationHeight
        WlrLayershell.keyboardFocus: WlrLayershell.None
        mask: Region {}
        color: "transparent"
        exclusionMode: ExclusionMode.Ignore
        Item {
            id: barContainer
            anchors.fill: parent

        }


        BackgroundEffect.blurRegion: Region {
            items: root.barItems
        }

        Process {
            id: outputQuery
            command: ["niri", "msg", "-j", "outputs"]

            stdout: StdioCollector {
                onStreamFinished: root.acceptOutputInfo(this.text)
            }

            onExited: (exitCode, exitStatus) => {
                if (exitCode !== 0 && !root.cavaStarted) {
                    console.warn(`cava-glass: output query exited with status ${exitCode}`)
                    root.startCava()
                }
            }
        }

        Process {
            id: cavaProcess
            command: [
                "sh", "-c",
                `printf "[general]\\nbars=${root.cavaBarCount}\\nframerate=${root.targetFramerate}\\nsensitivity=100\\n\\n[input]\\nmethod=pipewire\\nsource=auto\\n\\n[output]\\nmethod=raw\\nraw_target=/dev/stdout\\ndata_format=ascii\\nascii_max_range=100\\n" > /tmp/niri-cava-glass.conf && exec cava -p /tmp/niri-cava-glass.conf`
            ]

            stdout: SplitParser {
                onRead: data => {
                    const values = data.split(";")
                    for (const value of values) {
                        const parsed = parseInt(value.trim())
                        if (!isNaN(parsed)) root.acceptCavaValue(parsed)
                    }
                }
            }

            stderr: SplitParser {
                onRead: data => console.error(`cava-glass: ${data}`)
            }

            onRunningChanged: {
                if (!running) restartTimer.start()
            }
        }

        Timer {
            id: restartTimer
            interval: 2000
            onTriggered: cavaProcess.running = true
        }
    }

    Item {
        focus: true
        Keys.onEscapePressed: Qt.quit()
    }
}

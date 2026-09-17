import QtQuick
import QtQuick.Window
import Quickshell
import Quickshell.Io
import Quickshell.Wayland

// Renders the shadertoy shader in waves.frag on a fullscreen background layer surface.
ShellRoot {
    id: root

    // Qt 6 ShaderEffect only accepts preprocessed .qsb shaders; compile-shader.sh runs qsb on
    // waves.frag and the result is cached between runs.
    readonly property string stateDir: (Quickshell.env("XDG_STATE_HOME")
        || Quickshell.env("HOME") + "/.local/state") + "/quickshell/shaders"
    readonly property string scriptPath: Qt.resolvedUrl("compile-shader.sh").toString().replace(/^file:\/\//, "")
    readonly property string sourcePath: Qt.resolvedUrl("waves.frag").toString().replace(/^file:\/\//, "")
    readonly property string compiledPath: root.stateDir + "/niri-waves.frag.qsb"

    property string fragmentShaderUrl: ""

    Process {
        id: compiler
        running: true
        command: [root.scriptPath, root.sourcePath, root.compiledPath]

        stderr: StdioCollector {
            onStreamFinished: {
                if (text.trim() !== "") console.error(`shader-bg: ${text.trim()}`)
            }
        }

        onExited: (exitCode) => {
            if (exitCode !== 0) {
                console.error(`shader-bg: shader compilation failed with status ${exitCode}; no background shader`)
                return
            }
            // The query string busts Qt's shader cache when the file is recompiled.
            root.fragmentShaderUrl = "file://" + root.compiledPath + "?v=" + Date.now()
        }
    }

    Variants {
        model: Quickshell.screens

        PanelWindow {
            id: window
            required property var modelData
            screen: modelData

            WlrLayershell.namespace: "shader-background"
            WlrLayershell.layer: WlrLayershell.Background
            WlrLayershell.keyboardFocus: WlrLayershell.None

            anchors {
                top: true
                bottom: true
                left: true
                right: true
            }

            exclusionMode: ExclusionMode.Ignore
            color: "transparent"
            mask: Region {}

            ShaderEffect {
                id: waves
                anchors.fill: parent
                visible: root.fragmentShaderUrl !== ""
                fragmentShader: root.fragmentShaderUrl

                // Uniform block order must match waves.frag's std140 block.
                property real iTime: 0
                property vector2d iResolution: Qt.vector2d(
                    Math.round(width * Screen.devicePixelRatio),
                    Math.round(height * Screen.devicePixelRatio)
                )
                // Shadertoy only reads iMouse while a button is held; a background layer
                // surface never receives pointer input, so the camera keeps its framing.
                property vector4d iMouse: Qt.vector4d(0, 0, 0, 0)

                // Advances iTime one shadertoy-second per second, one tick per rendered frame.
                NumberAnimation on iTime {
                    from: 0
                    to: 100000
                    duration: 100000000
                    loops: Animation.Infinite
                    running: true
                }
            }
        }
    }
}

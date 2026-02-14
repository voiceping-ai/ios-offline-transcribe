// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "qwen-bench",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .executable(name: "qwen-bench", targets: ["qwen-bench"]),
    ],
    dependencies: [
        .package(path: "../../LocalPackages/QwenASRKit"),
    ],
    targets: [
        .executableTarget(
            name: "qwen-bench",
            dependencies: [
                .product(name: "QwenASRKit", package: "QwenASRKit"),
            ],
            linkerSettings: [
                .unsafeFlags([
                    "-L../../LocalPackages/SherpaOnnxKit/onnxruntime.xcframework/macos-arm64_x86_64",
                    "-lonnxruntime",
                ]),
                .linkedLibrary("c++"),
            ]
        ),
    ]
)

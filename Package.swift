// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "JapaneseRAGKit",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "JapaneseRAGKit", targets: ["JapaneseRAGKit"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.4"),
    ],
    targets: [
        .target(
            name: "JapaneseRAGKit",
            dependencies: [.product(name: "Tokenizers", package: "swift-transformers")]
        ),
        .executableTarget(name: "embed-check", dependencies: ["JapaneseRAGKit"]),
        .testTarget(name: "JapaneseRAGKitTests", dependencies: ["JapaneseRAGKit"]),
    ]
)

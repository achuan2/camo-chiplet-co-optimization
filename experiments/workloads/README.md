# Workload graphs

These ONNX computational graphs come from the preserved MOHaM benchmark distribution. Their hashes match `provenance/source-assets.json`.

Edge-3 uses BERT-Large, ResNet-50, and ResNet34-SSD1200, with `Conv_219` excluded from the last graph. Holdout-2 uses MobileNetV1-SSD300 and a U-Net-derived graph with `Conv_196` and `Conv_0` excluded. Configurations retain these exclusions; the processed U-Net-derived graph is not a complete end-to-end U-Net execution.


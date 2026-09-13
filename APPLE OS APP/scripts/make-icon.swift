// Renders the CADVERT app icon (the web app's logo mark: dark rounded square with a bold "C").
//   swift scripts/make-icon.swift CADVERT/Resources/Assets.xcassets/AppIcon.appiconset
import AppKit

let outDir = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "."
let fm = FileManager.default
try? fm.createDirectory(atPath: outDir, withIntermediateDirectories: true)

func render(size: CGFloat, rounded: Bool, inset: CGFloat) -> NSBitmapImageRep {
    let px = Int(size)
    let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: px, pixelsHigh: px, bitsPerSample: 8,
                               samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                               colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    rep.size = NSSize(width: size, height: size)   // 1 point == 1 pixel regardless of the screen scale
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    NSColor.clear.setFill()
    NSRect(x: 0, y: 0, width: size, height: size).fill()

    let rect = NSRect(x: inset, y: inset, width: size - 2 * inset, height: size - 2 * inset)
    let bg = NSColor(srgbRed: 0x1A/255.0, green: 0x1A/255.0, blue: 0x1A/255.0, alpha: 1)
    let path = rounded ? NSBezierPath(roundedRect: rect, xRadius: rect.width * 0.2237, yRadius: rect.width * 0.2237)
                       : NSBezierPath(rect: rect)
    bg.setFill(); path.fill()

    // Subtle top-light gradient like a pressed metal plate
    if let g = NSGradient(starting: NSColor.white.withAlphaComponent(0.08), ending: NSColor.clear) {
        NSGraphicsContext.saveGraphicsState(); path.addClip()
        g.draw(in: rect, angle: -90); NSGraphicsContext.restoreGraphicsState()
    }

    let font = NSFont.systemFont(ofSize: rect.width * 0.62, weight: .heavy)
    let attrs: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: NSColor.white]
    let s = NSAttributedString(string: "C", attributes: attrs)
    let sz = s.size()
    s.draw(at: NSPoint(x: rect.midX - sz.width / 2, y: rect.midY - sz.height / 2 + rect.height * 0.01))
    NSGraphicsContext.restoreGraphicsState()
    return rep
}

func writePNG(_ rep: NSBitmapImageRep, to path: String) {
    guard let png = rep.representation(using: .png, properties: [:]) else { fatalError("png") }
    try! png.write(to: URL(fileURLWithPath: path))
}

// iOS: full-bleed square (the system applies the mask). macOS: rounded with HIG margin.
writePNG(render(size: 1024, rounded: false, inset: 0), to: "\(outDir)/icon-ios-1024.png")
let macSizes: [Int] = [16, 32, 64, 128, 256, 512, 1024]
for s in macSizes {
    let inset = CGFloat(s) * 0.098
    writePNG(render(size: CGFloat(s), rounded: true, inset: inset), to: "\(outDir)/icon-mac-\(s).png")
}

var images: [[String: String]] = [
    ["filename": "icon-ios-1024.png", "idiom": "universal", "platform": "ios", "size": "1024x1024"],
]
let macEntries: [(Int, Int)] = [(16,1),(16,2),(32,1),(32,2),(128,1),(128,2),(256,1),(256,2),(512,1),(512,2)]
for (pt, scale) in macEntries {
    images.append(["filename": "icon-mac-\(pt * scale).png", "idiom": "mac", "scale": "\(scale)x", "size": "\(pt)x\(pt)"])
}
let contents: [String: Any] = ["images": images, "info": ["author": "xcode", "version": 1]]
let data = try! JSONSerialization.data(withJSONObject: contents, options: [.prettyPrinted, .sortedKeys])
try! data.write(to: URL(fileURLWithPath: "\(outDir)/Contents.json"))
print("wrote icons to \(outDir)")

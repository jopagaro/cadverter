// Compose App Store screenshots: place a window capture on a 2880x1800 canvas
// (one of Apple's accepted macOS sizes) with a neutral background.
//   swift make-screenshots.swift <out.png> <in.png> [caption]
import AppKit

let args = CommandLine.arguments
guard args.count >= 3 else { print("usage: make-screenshots.swift out.png in.png [caption]"); exit(1) }
let outPath = args[1], inPath = args[2]
let caption = args.count > 3 ? args[3] : ""

let W = 2880.0, H = 1800.0
guard let src = NSImage(contentsOfFile: inPath) else { print("cannot read \(inPath)"); exit(1) }

let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(W), pixelsHigh: Int(H),
                           bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                           colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
rep.size = NSSize(width: W, height: H)
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)

// Background: the app's own light surface tone, so the shot reads as one piece.
NSColor(srgbRed: 0xFA/255.0, green: 0xFA/255.0, blue: 0xF8/255.0, alpha: 1).setFill()
NSRect(x: 0, y: 0, width: W, height: H).fill()

let topInset = caption.isEmpty ? 90.0 : 230.0
let side = 120.0
let avail = NSRect(x: side, y: 90, width: W - side * 2, height: H - topInset - 90)
let s = src.size
let scale = min(avail.width / s.width, avail.height / s.height)
let drawn = NSRect(x: (W - s.width * scale) / 2,
                   y: avail.minY + (avail.height - s.height * scale) / 2,
                   width: s.width * scale, height: s.height * scale)

// Soft shadow so the window separates from the background.
let shadow = NSShadow()
shadow.shadowColor = NSColor.black.withAlphaComponent(0.18)
shadow.shadowBlurRadius = 40
shadow.shadowOffset = NSSize(width: 0, height: -12)
shadow.set()
NSBezierPath(roundedRect: drawn, xRadius: 18, yRadius: 18).fill()
NSShadow().set()

let clip = NSBezierPath(roundedRect: drawn, xRadius: 18, yRadius: 18)
NSGraphicsContext.saveGraphicsState(); clip.addClip()
src.draw(in: drawn, from: .zero, operation: .sourceOver, fraction: 1)
NSGraphicsContext.restoreGraphicsState()

if !caption.isEmpty {
    let style = NSMutableParagraphStyle(); style.alignment = .center
    let attrs: [NSAttributedString.Key: Any] = [
        .font: NSFont.systemFont(ofSize: 74, weight: .bold),
        .foregroundColor: NSColor(srgbRed: 0x1A/255.0, green: 0x1A/255.0, blue: 0x1A/255.0, alpha: 1),
        .paragraphStyle: style,
    ]
    let text = NSAttributedString(string: caption, attributes: attrs)
    let box = NSRect(x: side, y: H - topInset + 40, width: W - side * 2, height: 110)
    text.draw(in: box)
}

NSGraphicsContext.restoreGraphicsState()
guard let png = rep.representation(using: .png, properties: [:]) else { exit(1) }
try! png.write(to: URL(fileURLWithPath: outPath))
print("wrote \(outPath)")

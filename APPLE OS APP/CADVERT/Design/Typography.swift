import SwiftUI

/// Font tokens. The web app uses Satoshi at 13px base; on Apple platforms we use the
/// system font (SF Pro / SF Mono) which is the CSS fallback stack, with the same sizes.
/// iOS gets a small scale bump because touch targets and viewing distance differ.
enum Typo {
    static let scale: CGFloat = {
        #if os(iOS)
        return 1.15
        #else
        return 1.0
        #endif
    }()

    static func font(_ size: CGFloat, _ weight: Font.Weight = .medium, mono: Bool = false) -> Font {
        .system(size: size * scale, weight: weight, design: mono ? .monospaced : .default)
    }

    static func pt(_ size: CGFloat) -> CGFloat { size * scale }
}

extension View {
    /// Shorthand: `.typo(12, .semibold)` / `.typo(11, mono: true)`
    func typo(_ size: CGFloat, _ weight: Font.Weight = .medium, mono: Bool = false) -> some View {
        font(Typo.font(size, weight, mono: mono))
    }
}

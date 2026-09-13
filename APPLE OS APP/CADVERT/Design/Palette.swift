import SwiftUI

/// The web app's CSS theme variables, one struct per colour scheme.
/// Values are copied 1:1 from `src/cadvert/static/index.html`.
struct Palette: Equatable {
    var bg: Color
    var surface: Color
    var surfaceAlt: Color
    var border: Color
    var accent: Color
    var accentSoft: Color
    var text: Color
    var textMuted: Color
    var textDim: Color
    var success: Color
    var successBg: Color
    var danger: Color
    var warning: Color
    var warningBg: Color
    var logoBg: Color
    var userAvatarBg: Color
    var isDark: Bool

    static let light = Palette(
        bg:           Color(hex: 0xFAFAF8),
        surface:      Color(hex: 0xFFFFFF),
        surfaceAlt:   Color(hex: 0xF5F5F3),
        border:       Color(hex: 0xEBEBEA),
        accent:       Color(hex: 0x1A1A1A),
        accentSoft:   Color(hex: 0x1A1A1A).opacity(0.08),
        text:         Color(hex: 0x1A1A1A),
        textMuted:    Color(hex: 0x888888),
        textDim:      Color(hex: 0x999999),
        success:      Color(hex: 0x1B7A3D),
        successBg:    Color(hex: 0xE8F5EC),
        danger:       Color(hex: 0xDC3545),
        warning:      Color(hex: 0xB8860B),
        warningBg:    Color(hex: 0xB8860B).opacity(0.12),
        logoBg:       Color(hex: 0x1A1A1A),
        userAvatarBg: Color(hex: 0xF5F5F3),
        isDark: false
    )

    static let dark = Palette(
        bg:           Color(hex: 0x111116),
        surface:      Color(hex: 0x18181E),
        surfaceAlt:   Color.white.opacity(0.03),
        border:       Color(hex: 0x26262E),
        accent:       Color(hex: 0x5B6CFF),
        accentSoft:   Color(hex: 0x5B6CFF).opacity(0.08),
        text:         Color(hex: 0xE0E0E4),
        textMuted:    Color(hex: 0x666666),
        textDim:      Color(hex: 0x555555),
        success:      Color(hex: 0x2BB464),
        successBg:    Color(hex: 0x2BB464).opacity(0.12),
        danger:       Color(hex: 0xF87171),
        warning:      Color(hex: 0xFBBF24),
        warningBg:    Color(hex: 0xFBBF24).opacity(0.12),
        logoBg:       Color(hex: 0x5B6CFF),
        userAvatarBg: Color(hex: 0x26262E),
        isDark: true
    )

    static func forScheme(_ scheme: ColorScheme) -> Palette {
        scheme == .dark ? .dark : .light
    }
}

extension Color {
    init(hex: UInt32, alpha: Double = 1) {
        let r = Double((hex >> 16) & 0xFF) / 255
        let g = Double((hex >> 8) & 0xFF) / 255
        let b = Double(hex & 0xFF) / 255
        self.init(.sRGB, red: r, green: g, blue: b, opacity: alpha)
    }
}

private struct PaletteKey: EnvironmentKey {
    static let defaultValue: Palette = .light
}

extension EnvironmentValues {
    var palette: Palette {
        get { self[PaletteKey.self] }
        set { self[PaletteKey.self] = newValue }
    }
}

/// User-facing theme preference, mirroring the web app's Light / Dark pills
/// (plus "System", which the web app uses when nothing is saved).
enum ThemePreference: String, CaseIterable, Identifiable {
    case system, light, dark
    var id: String { rawValue }

    var colorScheme: ColorScheme? {
        switch self {
        case .system: return nil
        case .light:  return .light
        case .dark:   return .dark
        }
    }

    var label: String {
        switch self {
        case .system: return "System"
        case .light:  return "Light"
        case .dark:   return "Dark"
        }
    }
}

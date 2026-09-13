import SwiftUI

/// `.views-strip` — horizontal row of rendered view thumbnails (front/back/top/…/iso).
struct ViewsStrip: View {
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p

    var body: some View {
        Group {
            if let images = model.session?.images, !images.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 5) {
                        ForEach(images) { img in
                            ViewThumb(image: img)
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                }
            } else {
                HStack {
                    Text("Rendered views will appear here after upload")
                        .typo(11, .medium)
                        .foregroundStyle(p.textDim)
                    Spacer()
                }
                .padding(.horizontal, 16)
            }
        }
        .frame(height: 66)
        .background(p.surface)
    }
}

struct ViewThumb: View {
    var image: ViewImage
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p
    @State private var hover = false

    var body: some View {
        let url = model.imageURL(for: image)
        Button {
            if let url { model.lightboxURL = url }
        } label: {
            VStack(spacing: 0) {
                ZStack {
                    p.surfaceAlt
                    if let url {
                        AsyncImage(url: url) { phase in
                            switch phase {
                            case .success(let img): img.resizable().aspectRatio(contentMode: .fill)
                            case .failure: Image(systemName: "photo").foregroundStyle(p.textDim)
                            default: ProgressView().controlSize(.mini)
                            }
                        }
                    }
                }
                .frame(width: 56, height: 44)
                .clipped()
                Text(image.label.uppercased())
                    .font(.system(size: 7.5 * Typo.scale, weight: .medium))
                    .tracking(0.3)
                    .foregroundStyle(hover ? p.accent : p.textDim)
                    .padding(.horizontal, 4)
                    .padding(.vertical, 3)
                    .frame(width: 56)
                    .background(p.surfaceAlt)
            }
            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).stroke(hover ? p.accent : p.border, lineWidth: 1))
        }
        .buttonStyle(.plain)
        .onHover { hover = $0 }
        .help(image.label)
    }
}

/// `.lightbox` — full-screen image viewer, click anywhere to close.
struct LightboxView: View {
    var url: URL
    var dismiss: () -> Void
    @State private var zoom: CGFloat = 1

    var body: some View {
        ZStack {
            Color.black.opacity(0.8).ignoresSafeArea()
            GeometryReader { geo in
                AsyncImage(url: url) { phase in
                    if case .success(let img) = phase {
                        img.resizable().aspectRatio(contentMode: .fit)
                            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                    } else {
                        ProgressView().tint(.white)
                    }
                }
                .scaleEffect(zoom)
                .frame(width: geo.size.width * 0.9, height: geo.size.height * 0.9)
                .frame(width: geo.size.width, height: geo.size.height)
            }
            .gesture(MagnifyGesture().onChanged { zoom = max(1, min(4, $0.magnification)) }.onEnded { _ in zoom = 1 })
        }
        .contentShape(Rectangle())
        .onTapGesture { dismiss() }
        #if os(macOS)
        .onExitCommand { dismiss() }
        #endif
    }
}

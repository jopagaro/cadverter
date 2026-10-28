import SwiftUI

/// Settings → Acknowledgements: every open-source component inside the app, with its license.
/// The OpenCASCADE notice stays at the top because its license asks for it to be prominent.
struct AcknowledgementsView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p
    @State private var query = ""
    @State private var selection: Acknowledgements.Component.ID?
    private let acks = Acknowledgements.load()

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Acknowledgements").typo(13, .semibold).foregroundStyle(p.text)
                Spacer()
                CloseCircle { model.sheet = nil }
            }
            .padding(.horizontal, 20).padding(.vertical, 14)
            Hairline()
            Text("CADVERT is built on open-source software and makes use of facilities provided by the Open CASCADE Technology software. Each component is listed below with its license.")
                .typo(11, .medium).foregroundStyle(p.textMuted).lineSpacing(2)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 20).padding(.vertical, 12)
            Hairline()
            if let acks {
                content(acks)
            } else {
                Text("The acknowledgements file is missing from this build.")
                    .typo(11, .medium).foregroundStyle(p.danger)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .background(p.surface)
        #if os(macOS)
        .frame(width: 800, height: 580)
        #endif
    }

    @ViewBuilder
    private func content(_ acks: Acknowledgements) -> some View {
        #if os(macOS)
        HStack(spacing: 0) {
            VStack(spacing: 0) {
                searchField.padding(12)
                ScrollView { list(acks).padding(.horizontal, 8).padding(.bottom, 12) }
            }
            .frame(width: 260)
            Hairline(vertical: true)
            Group {
                if let c = acks.components.first(where: { $0.id == selection }) {
                    ComponentLicense(component: c)
                } else {
                    Text("Select a component").typo(11, .medium).foregroundStyle(p.textDim)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .onAppear { if selection == nil { selection = acks.components.first?.id } }
        #else
        NavigationStack {
            ScrollView {
                VStack(spacing: 12) { searchField; list(acks) }.padding(16)
            }
            .navigationDestination(item: $selection) { id in
                if let c = acks.components.first(where: { $0.id == id }) { ComponentLicense(component: c) }
            }
        }
        #endif
    }

    private var searchField: some View {
        TextField("Search components or licenses", text: $query).fieldChrome(mono: false)
            #if os(iOS)
            .textInputAutocapitalization(.never).autocorrectionDisabled()
            #endif
    }

    private func list(_ acks: Acknowledgements) -> some View {
        LazyVStack(alignment: .leading, spacing: 2) {
            ForEach(acks.sections(matching: query)) { section in
                SectionLabel(text: section.group)
                    .padding(.horizontal, 10).padding(.top, 12).padding(.bottom, 4)
                ForEach(section.items) { c in
                    Button { selection = c.id } label: {
                        VStack(alignment: .leading, spacing: 1) {
                            Text(c.name).typo(11.5, .semibold).foregroundStyle(p.text)
                            Text(c.license).typo(10, .medium).foregroundStyle(p.textDim).lineLimit(1)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 10).padding(.vertical, 6)
                        .background(selection == c.id ? p.accentSoft : .clear,
                                    in: RoundedRectangle(cornerRadius: 7, style: .continuous))
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }
}

/// One component: what it is, its license, any required notice, and the full license texts.
private struct ComponentLicense: View {
    let component: Acknowledgements.Component
    @Environment(\.palette) private var p

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text(component.name).typo(16, .bold).foregroundStyle(p.text)
                        if !component.version.isEmpty {
                            Text(component.version).typo(11, .medium, mono: true).foregroundStyle(p.textDim)
                        }
                    }
                    Text(component.license).typo(11.5, .medium).foregroundStyle(p.textMuted)
                    if !component.url.isEmpty, let url = URL(string: component.url) {
                        Link(component.url, destination: url).typo(11, .medium).foregroundStyle(p.accent)
                    }
                }
                if !component.notice.isEmpty {
                    Text(component.notice)
                        .typo(11.5, .medium).foregroundStyle(p.text).lineSpacing(2)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(p.surfaceAlt, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                }
                ForEach(Array(component.texts.enumerated()), id: \.offset) { _, text in
                    VStack(alignment: .leading, spacing: 6) {
                        SectionLabel(text: text.title)
                        Text(text.body)
                            .typo(10.5, .regular, mono: true).foregroundStyle(p.textMuted)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .id(component.id)   // start each component at the top
    }
}

import SwiftUI

/// `.input-bar` — multiline field + Send.
struct InputBar: View {
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p
    @FocusState private var focused: Bool

    var body: some View {
        @Bindable var model = model
        HStack(alignment: .bottom, spacing: 8) {
            TextField("Ask about this part...", text: $model.draft, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...5)
                .typo(12, .medium)
                .foregroundStyle(p.text)
                .padding(.horizontal, 12)
                .padding(.vertical, 9)
                .background(p.surfaceAlt, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).stroke(focused ? p.textMuted : p.border, lineWidth: 1))
                .focused($focused)
                .disabled(!model.canChat)
                .opacity(model.canChat ? 1 : 0.5)
                .onSubmit { model.sendDraft() }
                #if os(iOS)
                .submitLabel(.send)
                #endif

            Button("Send") { model.sendDraft() }
                .buttonStyle(PrimaryButtonStyle())
                .disabled(!model.canChat || model.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(p.surface)
        .onChange(of: model.session?.id) { _, new in
            // Auto-focus after upload like the web app — but only where a keyboard won't cover the UI.
            #if os(macOS)
            if new != nil { focused = true }
            #endif
        }
    }
}

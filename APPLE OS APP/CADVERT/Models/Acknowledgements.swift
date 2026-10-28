import Foundation

/// Every third-party component the app ships, with its license text.
/// Generated from the bundled engines by scripts/collect-licenses.py (`make licenses`) into
/// Resources/Acknowledgements.json; this only reads it.
struct Acknowledgements: Decodable {
    struct LicenseText: Decodable, Hashable {
        let title: String
        let body: String
    }

    struct Component: Decodable, Identifiable, Hashable {
        let id: String
        let name: String
        let version: String
        let license: String
        let url: String
        let group: String
        /// Attribution a license asks to be shown (e.g. OpenCASCADE, FreeType), or a short note.
        let notice: String
        let texts: [LicenseText]
    }

    struct Section: Identifiable {
        let group: String
        let items: [Component]
        var id: String { group }
    }

    let groups: [String]
    let components: [Component]

    static func load(from bundle: Bundle = .main) -> Acknowledgements? {
        guard let url = bundle.url(forResource: "Acknowledgements", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Acknowledgements.self, from: data)
    }

    /// Components in display order, grouped, keeping only those whose name or license matches.
    func sections(matching query: String = "") -> [Section] {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        let matches = q.isEmpty ? components : components.filter {
            $0.name.localizedCaseInsensitiveContains(q) || $0.license.localizedCaseInsensitiveContains(q)
        }
        return groups.compactMap { group in
            let items = matches.filter { $0.group == group }
            return items.isEmpty ? nil : Section(group: group, items: items)
        }
    }
}

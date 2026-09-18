#if os(macOS)
import Foundation
import Darwin

/// Runs `cadvert-server` (uvicorn) as a child process on a random loopback port.
/// This is what makes the Mac app a standalone, offline product: the same Python engine
/// that powers the web app, driven over HTTP on 127.0.0.1 and never exposed to the network.
final class LocalEngine {
    private var process: Process?
    private let logLock = NSLock()
    private var logBuffer = ""
    private(set) var baseURL: URL?

    var isRunning: Bool { process?.isRunning ?? false }

    func snapshotLog() -> String {
        logLock.lock(); defer { logLock.unlock() }
        return logBuffer
    }

    private func appendLog(_ s: String) {
        logLock.lock(); defer { logLock.unlock() }
        logBuffer += s
        if logBuffer.count > 200_000 { logBuffer = String(logBuffer.suffix(150_000)) }
    }

    /// Launches the engine and waits until `GET /config` answers.
    func start(python: URL, openAIKey: String?, anthropicKey: String? = nil, extraEnvironment: [String: String] = [:], readyTimeout: TimeInterval = 120) async throws -> URL {
        stop()
        let port = Self.freePort()
        let base = URL(string: "http://127.0.0.1:\(port)")!

        let proc = Process()
        proc.executableURL = python
        // Bootstrap: run uvicorn in-process and exit the moment the app (our parent) is gone,
        // so a crashed or force-quit app never leaves an orphaned engine listening.
        proc.arguments = ["-u", "-c", Self.bootstrapScript, String(getpid()), String(port)]

        var env = ProcessInfo.processInfo.environment
        env["DISABLE_AUTH"] = "1"                 // the app itself is the paywall; no Google sign-in locally
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        env["VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN"] = "1"
        // Cached parts belong in Caches: everything there is regenerable from the user's
        // own CAD file, and the system may reclaim it under disk pressure. Under the App
        // Sandbox this resolves inside the app container automatically.
        if let cache = LocalEngine.cacheDirectory {
            env["CADVERT_DATA_DIR"] = cache.path
        }
        env["MAX_FILE_MB"] = env["MAX_FILE_MB"] ?? "500"
        env["SESSION_TTL_HOURS"] = env["SESSION_TTL_HOURS"] ?? "72"
        env["ALLOWED_ORIGINS"] = "http://127.0.0.1"
        if let openAIKey, !openAIKey.isEmpty { env["OPENAI_API_KEY"] = openAIKey } else { env.removeValue(forKey: "OPENAI_API_KEY") }
        if let anthropicKey, !anthropicKey.isEmpty { env["ANTHROPIC_API_KEY"] = anthropicKey } else { env.removeValue(forKey: "ANTHROPIC_API_KEY") }
        for (k, v) in extraEnvironment { env[k] = v }
        proc.environment = env
        proc.currentDirectoryURL = FileManager.default.temporaryDirectory

        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] h in
            let d = h.availableData
            if d.isEmpty { return }
            self?.appendLog(String(decoding: d, as: UTF8.self))
        }
        proc.terminationHandler = { [weak self] p in
            self?.appendLog("\n[engine exited with status \(p.terminationStatus)]\n")
        }

        appendLog("▸ \(python.path) (uvicorn cadvert.server:app --host 127.0.0.1 --port \(port), watching parent pid \(getpid()))\n")
        do {
            try proc.run()
        } catch {
            throw CadvertError.engine("Could not launch the engine: \(error.localizedDescription)")
        }
        process = proc

        // Poll /config until the server answers.
        let client = CadvertClient(baseURL: base)
        let deadline = Date().addingTimeInterval(readyTimeout)
        while Date() < deadline {
            if !proc.isRunning {
                throw CadvertError.engine("The engine exited during startup.\n\n\(Self.tail(snapshotLog()))")
            }
            if (try? await client.fetchConfig()) != nil {
                baseURL = base
                appendLog("▸ engine ready on \(base.absoluteString)\n")
                return base
            }
            try? await Task.sleep(nanoseconds: 400_000_000)
        }
        stop()
        throw CadvertError.engine("The engine did not become ready within \(Int(readyTimeout))s.\n\n\(Self.tail(snapshotLog()))")
    }

    func stop() {
        guard let proc = process else { return }
        if proc.isRunning {
            proc.terminate()
            // uvicorn handles SIGTERM gracefully; give it a moment, then force.
            let deadline = Date().addingTimeInterval(3)
            while proc.isRunning && Date() < deadline { usleep(50_000) }
            if proc.isRunning { kill(proc.processIdentifier, SIGKILL) }
        }
        (proc.standardOutput as? Pipe)?.fileHandleForReading.readabilityHandler = nil
        process = nil
        baseURL = nil
    }

    deinit { stop() }

    // MARK: Helpers

    /// Runs the server and polls `getppid()`; when the app dies the parent changes (to launchd) and we exit.
    static let bootstrapScript = """
    import os, sys, threading, time
    parent, port = int(sys.argv[1]), int(sys.argv[2])
    def _watch():
        while True:
            time.sleep(1.5)
            if os.getppid() != parent:
                os._exit(0)
    threading.Thread(target=_watch, daemon=True).start()
    import uvicorn
    uvicorn.run("cadvert.server:app", host="127.0.0.1", port=port, log_level="info", access_log=False)
    """

    static func tail(_ s: String, lines: Int = 25) -> String {
        s.split(separator: "\n", omittingEmptySubsequences: false).suffix(lines).joined(separator: "\n")
    }

    /// Where cached parts live: `~/Library/Caches/<bundle id>/parts`, or inside the
    /// sandbox container when sandboxed. Created on demand.
    static var cacheDirectory: URL? {
        let fm = FileManager.default
        guard let base = fm.urls(for: .cachesDirectory, in: .userDomainMask).first else { return nil }
        let id = Bundle.main.bundleIdentifier ?? "com.cadvert.CADVERT"
        let dir = base.appendingPathComponent(id, isDirectory: true)
            .appendingPathComponent("parts", isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Ask the kernel for an unused loopback TCP port.
    static func freePort() -> Int {
        let sock = socket(AF_INET, SOCK_STREAM, 0)
        guard sock >= 0 else { return Int.random(in: 49152...65535) }
        defer { close(sock) }
        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        addr.sin_port = 0
        let bindResult = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(sock, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
        }
        guard bindResult == 0 else { return Int.random(in: 49152...65535) }
        var len = socklen_t(MemoryLayout<sockaddr_in>.size)
        let nameResult = withUnsafeMutablePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(sock, $0, &len) }
        }
        guard nameResult == 0 else { return Int.random(in: 49152...65535) }
        return Int(UInt16(bigEndian: addr.sin_port))
    }
}

/// Finds a Python interpreter that has the cadvert engine installed.
enum EngineLocator {
    /// The architecture this app slice is running as — the engine must match it.
    /// A universal build running natively on Apple silicon reports arm64; the same build
    /// running under Rosetta reports x86_64, and picks the Intel engine accordingly.
    static var currentArch: String {
        #if arch(arm64)
        return "arm64"
        #elseif arch(x86_64)
        return "x86_64"
        #else
        return "unknown"
        #endif
    }

    /// `CADVERT.app/Contents/Resources/engine-<arch>/bin/python3`, produced by
    /// scripts/bundle-engine.sh. Falls back to the legacy single-arch `engine/` folder.
    static func bundledPython() -> URL? {
        guard let res = Bundle.main.resourceURL else { return nil }
        let candidates = [
            res.appendingPathComponent("engine-\(currentArch)/bin/python3"),
            res.appendingPathComponent("engine/bin/python3"),
        ]
        return candidates.first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }

    /// True when the bundle ships an engine, but for a different architecture than this slice.
    /// Lets the UI say "this build has no Intel engine" instead of a generic failure.
    static func bundleHasOtherArchEngineOnly() -> Bool {
        guard let res = Bundle.main.resourceURL, bundledPython() == nil else { return false }
        let fm = FileManager.default
        let names = (try? fm.contentsOfDirectory(atPath: res.path)) ?? []
        return names.contains { $0.hasPrefix("engine-") }
    }

    static func candidates(customPath: String) -> [URL] {
        var list: [URL] = []
        let custom = customPath.trimmingCharacters(in: .whitespacesAndNewlines)
        if !custom.isEmpty {
            let expanded = (custom as NSString).expandingTildeInPath
            var url = URL(fileURLWithPath: expanded)
            // Accept a venv folder as well as a python binary.
            if FileManager.default.fileExists(atPath: url.appendingPathComponent("bin/python3").path) {
                url = url.appendingPathComponent("bin/python3")
            } else if FileManager.default.fileExists(atPath: url.appendingPathComponent("bin/python").path) {
                url = url.appendingPathComponent("bin/python")
            }
            list.append(url)
        }
        if let b = bundledPython() { list.append(b) }
        let home = FileManager.default.homeDirectoryForCurrentUser
        list.append(home.appendingPathComponent(".cadvert/engine/bin/python3"))
        list.append(home.appendingPathComponent(".cadvert/venv/bin/python3"))
        return list
    }

    /// Runs `python -c "import cadvert.server, uvicorn"` with a timeout.
    static func validate(_ python: URL) async -> (ok: Bool, detail: String) {
        guard FileManager.default.isExecutableFile(atPath: python.path) else {
            return (false, "not found: \(python.path)")
        }
        return await withCheckedContinuation { cont in
            let proc = Process()
            proc.executableURL = python
            proc.arguments = ["-c", "import cadvert, cadvert.server, uvicorn; print(getattr(cadvert, '__version__', 'ok'))"]
            var env = ProcessInfo.processInfo.environment
            env["PYTHONNOUSERSITE"] = "1"
            proc.environment = env
            let out = Pipe(); proc.standardOutput = out; proc.standardError = out
            var finished = false
            let lock = NSLock()
            proc.terminationHandler = { p in
                let text = String(decoding: out.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                lock.lock(); defer { lock.unlock() }
                if finished { return }
                finished = true
                cont.resume(returning: (p.terminationStatus == 0, text.isEmpty ? "exit \(p.terminationStatus)" : text))
            }
            do { try proc.run() } catch {
                lock.lock(); defer { lock.unlock() }
                if !finished { finished = true; cont.resume(returning: (false, error.localizedDescription)) }
                return
            }
            DispatchQueue.global().asyncAfter(deadline: .now() + 45) {
                lock.lock(); let done = finished; lock.unlock()
                if !done, proc.isRunning { proc.terminate() }
            }
        }
    }

    /// First candidate that imports the engine, plus a human-readable trail for Settings.
    static func resolve(customPath: String) async -> (python: URL?, detail: String) {
        var trail: [String] = []
        for c in candidates(customPath: customPath) {
            let r = await validate(c)
            trail.append("\(r.ok ? "✓" : "✗") \(c.path) — \(r.detail)")
            if r.ok { return (c, trail.joined(separator: "\n")) }
        }
        if trail.isEmpty { trail.append("no candidates") }
        return (nil, trail.joined(separator: "\n"))
    }
}
#endif

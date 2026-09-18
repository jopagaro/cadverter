# CADVERT Apple app — common tasks
XCODEPROJ = CADVERT.xcodeproj
SCHEME    = CADVERT
BUILD     = build

.PHONY: generate icon build-mac build-ios test run-mac engine engine-intel engine-all refresh \
        archive-mac archive-universal release-arm64 release-intel release-both dmg clean

generate:            ## Regenerate CADVERT.xcodeproj from project.yml
	xcodegen generate

icon:                ## Render the app icon set
	swift scripts/make-icon.swift CADVERT/Resources/Assets.xcassets/AppIcon.appiconset

build-mac: generate  ## Debug build for this Mac
	xcodebuild -project $(XCODEPROJ) -scheme $(SCHEME) -destination 'platform=macOS' -configuration Debug \
	  -derivedDataPath $(BUILD) build | tail -5

build-ios: generate  ## Debug build for the iPad simulator
	xcodebuild -project $(XCODEPROJ) -scheme $(SCHEME) -destination 'generic/platform=iOS Simulator' -configuration Debug \
	  -derivedDataPath $(BUILD) CODE_SIGNING_ALLOWED=NO build | tail -5

test: generate       ## Unit tests (macOS)
	xcodebuild -project $(XCODEPROJ) -scheme $(SCHEME) -destination 'platform=macOS' -derivedDataPath $(BUILD) test | tail -20

run-mac: build-mac   ## Launch the Debug Mac build
	open $(BUILD)/Build/Products/Debug/CADVERT.app

engine:              ## Build the bundled Python engine for this Mac's architecture
	./scripts/bundle-engine.sh --arch $(shell uname -m) --source ..

engine-intel:        ## Build the Intel (x86_64) engine — needs Rosetta on Apple silicon
	./scripts/bundle-engine.sh --arch x86_64 --source ..

engine-all: engine engine-intel  ## Build both engines (needed for a universal build)

refresh:             ## Reinstall current cadvert into existing engines (run before archiving)
	./scripts/refresh-engine.sh

archive-mac: generate ## Release archive for this Mac's architecture only
	CADVERT_ENGINE_ARCHS=$(shell uname -m) xcodebuild -project $(XCODEPROJ) -scheme $(SCHEME) \
	  -destination 'generic/platform=macOS' -configuration Release ARCHS=$(shell uname -m) ONLY_ACTIVE_ARCH=NO \
	  -archivePath $(BUILD)/CADVERT.xcarchive archive | tail -5

archive-universal: generate ## Universal archive carrying BOTH engines (~1.6 GB — for the App Store)
	CADVERT_ENGINE_ARCHS="arm64 x86_64" xcodebuild -project $(XCODEPROJ) -scheme $(SCHEME) \
	  -destination 'generic/platform=macOS' -configuration Release ARCHS="arm64 x86_64" ONLY_ACTIVE_ARCH=NO \
	  -archivePath $(BUILD)/CADVERT-universal.xcarchive archive | tail -5

# ── Direct-sale builds: one DMG per architecture keeps each download ~800 MB ──
release-arm64: generate  ## Apple silicon .app + .dmg
	./scripts/build-release.sh --arch arm64

release-intel: generate  ## Intel .app + .dmg
	./scripts/build-release.sh --arch x86_64

release-both: release-arm64 release-intel  ## Both per-architecture DMGs

dmg:                 ## DMG from an exported app at build/export/CADVERT.app
	./scripts/make-dmg.sh $(BUILD)/export/CADVERT.app $(BUILD)/CADVERT.dmg

clean:
	rm -rf $(BUILD)

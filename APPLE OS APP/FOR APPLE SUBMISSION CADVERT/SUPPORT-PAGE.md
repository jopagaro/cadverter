# Support page (draft)

Apple requires a support URL. It can be a single page; it just has to exist, be reachable,
and give people a way to reach you.

---

# CADVERT Support

## Getting started

Open a CAD file by dragging it onto the window or pressing ⌘O. CADVERT reads STEP (.step,
.stp), IGES (.iges, .igs), BREP, STL and OBJ files.

STEP, IGES and BREP get the full analysis — detected features, exact measurements and
spatial relationships. STL and OBJ are triangle meshes with no exact geometry to read, so
those report size and shape only. If you need full analysis, export STEP from your CAD tool.

## Asking questions about a part

Analysis itself needs no AI and no setup. To ask questions, choose a provider in Settings:

**Apple Intelligence** runs on your Mac, costs nothing and needs no key. It requires an
Apple silicon Mac running macOS 26 with Apple Intelligence turned on.

**OpenAI or Anthropic** need an API key from that provider, billed to you at their rates.
Your key is stored in your Mac's Keychain and sent only to them.

## Common questions

**Why does it say Apple Intelligence is unavailable?**
Three possible reasons, and the app tells you which. Intel Macs cannot run it at any macOS
version. On Apple silicon it may be switched off in System Settings → Apple Intelligence &
Siri, or the model may still be downloading. In every case you can add your own API key
instead, and file analysis works regardless.

**The app is large. Why?**
CADVERT contains a complete geometry kernel — the same class of engine professional CAD
systems use. That is what lets it work offline with no server and report exact values
rather than estimates.

**Does my CAD file leave my Mac?**
No. Analysis happens entirely on your machine. If you use OpenAI or Anthropic for
questions, a text summary of the geometry is sent to that provider so they can answer;
the file itself is never uploaded anywhere.

**How do I free up disk space?**
Settings → Storage shows how much cached analysis is stored and has a Clear button.
Everything cached rebuilds from your original file, so clearing loses nothing.

**A large assembly takes a while.**
Analysis time scales with the number of faces, not file size. A few thousand faces takes
under a minute. It runs on one core, so the machine stays responsive.

**Can I use it on Windows?**
Not yet. A Windows version is planned.

## Contact

[your email] — please include the macOS version, whether your Mac is Apple silicon or
Intel, and the file type you were working with.

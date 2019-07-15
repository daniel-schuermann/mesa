---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**Screenshots**
For rendering errors, please insert screenshots showing incorrect rendering (ACO) and correct rendering (LLVM).

**RenderDoc capture:**
Please try to link to a RenderDoc capture of the issue (or scene were ACO is slower than LLVM), using either the ACO or LLVM compiler for this is fine. Once RenderDoc is installed, a capture can be created by:

- setting the environment variable ENABLE_VULKAN_RENDERDOC_CAPTURE to "1"
- pressing F12 once the problem is visible (or for hangs, would be visible if ACO was used)
- uploading the .rdc file created in /tmp/renderdoc

For Steam, the ENABLE_VULKAN_RENDERDOC_CAPTURE can be set by setting the game's launch options to "ENABLE_VULKAN_RENDERDOC_CAPTURE=1 %command%", possibly including "RADV_PERFTEST=llvm" at the beginning if you want LLVM to be used. You'll want to change the launch options back to what they were before once you've finished.

An alternative method is to close Steam and start it from the console via "ENABLE_VULKAN_RENDERDOC_CAPTURE=1 steam" and run the game.

**System information:**
 - GPU: [e.g. Radeon RX480]
 - ACO version: [either git commit/branch or package version]

**Additional context**
Add any other context about the problem here.

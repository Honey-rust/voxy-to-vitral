# Reproducing the verified v41 terrain integration

Verified on 2026-10-14 against the working leaf-bounds-v41 Fabric JARs.

The current source already includes the v41 provider API, CPU traversal, Sodium draw hooks,
terrain depth and visibility handling, water rendering, zoom response, solid leaf bounds,
and the cache/upload optimizations. No manual replacement of classes inside a JAR is needed.

## Build

Use JDK 25 and the checked-in Gradle Wrapper. In PowerShell:

```powershell
# Vitrail: Fabric and NeoForge artifacts for Minecraft 26.2
.\gradlew.bat clean build --no-build-cache

# Vitrail: also verify Minecraft 26.3
.\gradlew.bat clean build --no-build-cache '-Pminecraft=26.3'

# Voxy: Minecraft 26.2 only, in the Voxy repository
.\gradlew.bat clean build --no-build-cache
```

All three builds passed. Vitrail's existing tests passed for both game versions.
Voxy currently has no test sources. The previous source/resource selection fixes and
the existing warning policy remain enabled.

Vitrail's Fabric artifact is under `fabric/build/mc26.2/libs/`; the combined-loader
artifact is under `build/libs/`. Voxy's artifact is under `build/libs/`; its filename
may contain the Git commit. Version, commit and build timestamp metadata can differ
from the historical leaf-bounds-v41 filename without changing the terrain implementation.

## Artifact comparison

The clean-built 26.2 artifacts were compared against the working reference JARs:

- Vitrail: all 767 `dev/vitrail/` classes present.
- Voxy: all 406 `me/cortex/voxy/` classes present.
- Classes with different bytes had identical method signatures, executable instructions,
  constants and runtime annotations when inspected with `javap`.
- Mixin configuration settings and entry sets match. Only the order of the two added
  compatibility mixins differs in Vitrail's main configuration.
- Shader assets, access wideners and provider entry points are retained.

Different debug information, archive metadata or build stamps mean identical SHA-256
hashes are not expected. The check establishes implementation correspondence, not a
replacement for testing the newly built artifacts in game.

## Comparing in game

Use the same Minecraft instance, server address, player, world and LOD database for
both artifact pairs. Keep the working reference JARs backed up. Install exactly one
Vitrail JAR and one Voxy JAR after exiting Minecraft.

Voxy selects multiplayer storage under `.voxy/saves/<server address>/`, replacing
the address colon with an underscore. Connecting through a different hostname, IP
or port therefore selects a different database, even when it reaches the same world.
A fresh database cannot immediately render terrain visited only in the old database.

If there is no distant landscape, first check provider registration, aggregate upload
and draw messages in the log, then confirm the selected database contains distant terrain.
Do not merge databases belonging to different worlds.

The new artifacts have not yet been compared interactively in the game during this
verification session. No GitHub branch was pushed and the working reference mods were
not replaced.


# proc_solar — Procedural Solar System (Rust software renderer)

Render the Sun and **8 unique planets** (plus a moon and rings) using **ONLY procedural color shaders** — no textures or PBR materials.
Everything is computed in a CPU software renderer via ray–sphere and ray–ring intersections.

## Features (scoring checklist)
- ✅ Star (Sun) with 4+ shader layers (core, granulation, radial falloff, flares).
- ✅ Rocky planet (cratered), rocky desert, volcanic, and an **ocean world with clouds**.
- ✅ Gas giant with **bands + great spot**.
- ✅ Gas giant with **ring system** (separate ring model, fully procedural).
- ✅ Ice giants (soft bands and turbulent).
- ✅ **Moon** orbiting the ocean world (separate sphere).  
- ✅ Optional rotation via `--time` parameter (affects band drift and UV rotation).
- ✅ No textures or materials — strictly color functions.

## Build
```bash
# recommended: Rust 1.73+
cargo build --release
```

## Quick run (render everything)
```bash
cargo run --release -- --body all --out out --width 1400 --height 900 --hq
```

This writes PNGs to the `out/` folder:
- `sun.png`
- `rockycratered.png`
- `rockydesert.png`
- `oceanworld.png` (the system render also shows a small moon)
- `volcanic.png`
- `gasbands.png`
- `gasrings.png` (+ rings)
- `icesoft.png`
- `iceturbulent.png`
- `system.png` (if you render `--body system`)

## Render a single body
```bash
# e.g., Jupiter-like gas giant
cargo run --release -- --body gasbands --hq

# e.g., ringed giant
cargo run --release -- --body gasrings --hq

# the full system (sun + 8 planets + moon + rings)
cargo run --release -- --body system --width 1600 --height 900 --hq
```

## Optional animation
Use `--time` to offset band drift and rotation (use in a loop to render frames).
```bash
cargo run --release -- --body system --time 3.5
```

## Notes for grading
- **Layers** per shader: most bodies mix 4+ layers (base palette, bands/dunes/continents, fbm detail, highlights/clouds/spots).  
- **Rings:** procedural bands, gaps & alpha, separate `Ring` model (20 pts).
- **Moon:** small sphere bound to ocean world (20 pts).
- **Creativity:** palettes & structures are configurable; inspect the shader functions and tweak parameters.
- **No textures/materials** used; all color comes from analytic functions (fbm, sine bands, craters via angular masks).

## Screenshots (add yours)
Place your renders here for the README submission:
```
out/sun.png
out/rockycratered.png
out/rockydesert.png
out/oceanworld.png
out/volcanic.png
out/gasbands.png
out/gasrings.png
out/icesoft.png
out/iceturbulent.png
out/system.png
```

---

### Folder structure
```
proc_solar/
  Cargo.toml
  src/
    main.rs
  out/ (created on first run)
```

## Window (interactive) — todo junto como sistema solar
Para ver todo en una **ventana** con animación, luz punto desde el **Sol** y órbitas circulares:

```bash
cargo run --release -- --window --fps 30 --width 1280 --height 800
```

- **ESC** para salir.
- `--orbits=false` si prefieres posiciones estáticas.
- La iluminación ahora sale **desde la posición del Sol** (sombras suaves con shadow ray).

> El modo ventana usa el mismo renderer CPU y shaders procedurales; sólo dibuja a un `minifb` en tiempo real.

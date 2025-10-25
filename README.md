# proc_solar — Renderizador procedural (Rust)
Sistema solar procedimental con **sombras**, **luz puntual** desde el **Sol**, **8 planetas**, **anillos**, **luna para el planeta rocoso**, vista en **ventana interactiva** y **render a PNG**. Sin texturas ni materiales: todo es **shader de color**.

> Proyecto listo para el laboratorio: estrella (Sol), planeta rocoso, gigante gaseoso; más extras: anillos, luna, rotación y traslación.

---

## 🎯 Objetivos del lab cubiertos
- **Estrella (Sol):** `Body::Sun` con shader de convección/flares.
- **Planeta rocoso:** `Body::RockyCratered` (cráteres por capas). **Luna implementada** (modelo separado).
- **Gigante gaseoso:** `Body::GasBands` y `Body::GasRings` (con sistema de **anillos**).
- **Solo shaders (sin texturas):** patrones generados por ruido, fbm, bandas, máscaras y blends.
- **Extras para puntos:**
  - **+20 pts** anillos (gaseoso).
  - **+20 pts** luna (rocoso).
  - **+10/20/30 pts** planetas extra (hay varios más allá de los 3 requeridos).
  - **Rotación y traslación** (opcional del lab) incluidas.

---

## 📦 Requisitos
- **Rust** estable y **cargo** (https://www.rust-lang.org/)
- El resto de dependencias se descargan automáticamente vía **Cargo**.

---

## 🚀 Ejecutar (ventana interactiva)
Sistema completo:
```bash
cargo run --release -- --window --width 1280 --height 800

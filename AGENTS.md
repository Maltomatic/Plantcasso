# Plantcasso — Agent Guidelines

## C++ Guidelines (ESP32 / Embedded)

### Language & Toolchain
- **Use C++23** (`-std=gnu++23`). Target is the **ESP32** (Xtensa LX6/LX7 or RISC-V on
  -C3/-S3 variants). Use the ESP-IDF toolchain (GCC) or Arduino-ESP32 core built on it.
- The ESP32 has a **single-precision FPU only** (no hardware `double`). Always prefer
  `float` and `float`-suffixed literals (`1.0f`). Avoid `double` in hot paths — it forces
  slow software emulation.
- Enable warnings: `-Wall -Wextra -Wshadow -Wconversion`.

### Memory: no surprise heap allocation
- **No dynamic allocation in steady-state / real-time paths.** Avoid `new`, `malloc`,
  `std::vector`, `std::string`, `std::map`, etc. on hot paths. Heap fragmentation on a
  long-running MCU leads to failures.
- Size buffers at compile time. Prefer `std::array<T, N>` over C arrays and over `std::vector`.
- Use **templates with compile-time dimensions** (e.g. `KMeans<N, DIM, K>`) so sizes are
  known to the compiler and storage can live in `.bss`/`.data` or on the stack.
- Place large objects in `static`/global storage (`.bss`) rather than on the stack — the
  ESP32 default task stack is small (~8 KB). Watch stack depth in recursive/deep calls.
- For caller-provided memory, accept a `std::span<T>` or a fixed pool; never allocate
  internally.

### Modern C++ for embedded (prefer these)
- `constexpr` / `consteval` — push computation to compile time; build lookup tables at
  compile time instead of runtime.
- `std::array`, `std::span` (C++20), `std::string_view` — zero-overhead, no allocation.
- `enum class` for type-safe states; `[[nodiscard]]`, `[[likely]]`/`[[unlikely]]`.
- Strong typing over raw ints (units, pin numbers). Use `std::int32_t`, `std::uint16_t`
  etc. from `<cstdint>` — never assume `int` width.
- `if constexpr` for compile-time branching across board variants.
- RAII for hardware resources (GPIO, I2C/SPI handles, mutexes) — acquire in ctor, release
  in dtor.
- `std::optional` / `std::expected` (C++23) for error handling instead of magic return
  values; avoid heavyweight machinery on hot paths.

### Avoid / use with care
- **Exceptions and RTTI**: typically disabled on ESP-IDF (`-fno-exceptions -fno-rtti`) to
  save flash/RAM. Don't write code that depends on them. Use `std::expected`/error codes.
- **`<iostream>`**: pulls in large code; use `printf`/ESP-IDF `ESP_LOGx` for logging.
- Recursion and unbounded loops in ISRs. Keep ISRs tiny; defer work to tasks/queues.
- `double`, `std::function` (heap-allocating), and virtual dispatch in hot paths.

### Style
- Squared Euclidean distance instead of `sqrt` when only comparing magnitudes.
- Deterministic, seedable RNG (small LCG) instead of `<random>` global state.
- Keep numerical work in `float`; use fixed-point integer math on FPU-less targets.

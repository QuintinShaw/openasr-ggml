#pragma once

#if defined(__linux__)
#include <charconv>
#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>

// Read the kernel's available-memory estimate, as OpenASR's Linux host probe does.
static std::optional<uint64_t> ggml_linux_mem_available() try {
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo) {
        return std::nullopt;
    }

    std::string line;
    while (std::getline(meminfo, line)) {
        constexpr char prefix[] = "MemAvailable:";
        if (line.compare(0, sizeof(prefix) - 1, prefix) != 0) {
            continue;
        }
        std::istringstream fields(line.substr(sizeof(prefix) - 1));
        std::string first;
        if (!(fields >> first)) {
            continue;
        }
        const char * begin = first.data();
        const char * end = begin + first.size();
        uint64_t kb;
        const auto parsed = std::from_chars(begin, end, kb);
        if (parsed.ec != std::errc{} || parsed.ptr != end) {
            continue;
        }
        constexpr uint64_t max = std::numeric_limits<uint64_t>::max();
        return kb > max / 1024 ? max : kb * 1024;
    }
    return std::nullopt;
} catch (const std::exception &) {
    return std::nullopt;
}
#endif

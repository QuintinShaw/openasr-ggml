#include "ggml-backend-dl.h"

#ifdef _WIN32

dl_handle * dl_load_library(const fs::path & path, const std::vector<fs::path> & dependency_dirs) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    // Every production caller supplies a verified absolute path. Resolve
    // dependencies only from the plugin's own directory, the application
    // directory (ggml-base/ggml), and System32. In particular, do not consult
    // the current directory or PATH, which would let an unrelated DLL shadow a
    // signed pack dependency.
    std::vector<DLL_DIRECTORY_COOKIE> dependency_cookies;
    dependency_cookies.reserve(dependency_dirs.size());
    bool dependency_setup_ok = true;
    for (const fs::path & dependency_dir : dependency_dirs) {
        DLL_DIRECTORY_COOKIE cookie = AddDllDirectory(dependency_dir.wstring().c_str());
        if (cookie == nullptr) {
            dependency_setup_ok = false;
            break;
        }
        dependency_cookies.push_back(cookie);
    }

    const DWORD flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                        LOAD_LIBRARY_SEARCH_APPLICATION_DIR |
                        LOAD_LIBRARY_SEARCH_SYSTEM32 |
                        LOAD_LIBRARY_SEARCH_USER_DIRS;
    HMODULE handle = dependency_setup_ok ? LoadLibraryExW(path.wstring().c_str(), nullptr, flags) : nullptr;

    for (auto iterator = dependency_cookies.rbegin(); iterator != dependency_cookies.rend(); ++iterator) {
        RemoveDllDirectory(*iterator);
    }

    SetErrorMode(old_mode);

    return handle;
}

dl_handle * dl_load_library(const fs::path & path) {
    return dl_load_library(path, {});
}

void * dl_get_sym(dl_handle * handle, const char * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    void * p = (void *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    return p;
}

const char * dl_error() {
    return "";
}

#else

dl_handle * dl_load_library(const fs::path & path, const std::vector<fs::path> & dependency_dirs) {
    (void) dependency_dirs;
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    return handle;
}

dl_handle * dl_load_library(const fs::path & path) {
    return dl_load_library(path, {});
}

void * dl_get_sym(dl_handle * handle, const char * name) {
    return dlsym(handle, name);
}

const char * dl_error() {
    const char *rslt = dlerror();
    return rslt != nullptr ? rslt : "";
}

#endif

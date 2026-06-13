#include "ggml-backend-dl.h"

#ifdef _WIN32

dl_handle * dl_load_library(const fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    // LOAD_WITH_ALTERED_SEARCH_PATH: when `path` is absolute (it is for downloaded
    // GPU packs under OPENASR_HOME/backends/<vendor>/<version>/), resolve the
    // plugin's dependency chain starting from the plugin's OWN directory instead
    // of the host exe directory. The GPU plugins (ggml-hip / ggml-vulkan /
    // ggml-cuda) link satellite runtime DLLs (amdhip64, rocblas, hipblas,
    // vulkan-1, cudart, cublas, ...) that are staged next to the plugin in its
    // pack dir — a non-exe directory the default search order never visits. With
    // a plain LoadLibraryW those imports go unresolved, the load returns NULL, and
    // the backend is silently never registered (the engine then fails open to
    // CPU). For base plugins loaded by bare filename next to the exe, the flag is
    // a no-op (it only alters the search for absolute paths), so CPU loading is
    // unchanged.
    HMODULE handle = LoadLibraryExW(path.wstring().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);

    SetErrorMode(old_mode);

    return handle;
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

dl_handle * dl_load_library(const fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    return handle;
}

void * dl_get_sym(dl_handle * handle, const char * name) {
    return dlsym(handle, name);
}

const char * dl_error() {
    const char *rslt = dlerror();
    return rslt != nullptr ? rslt : "";
}

#endif

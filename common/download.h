#pragma once

#include "hf-cache.h"

#include <string>
#include <utility>
#include <vector>

// Minimal port of llama.cpp's downloader (common/download.{h,cpp}), limited to
// the functionality needed for the -hf command line option.

// split HF repo with tag into <repo, tag>, for example:
// - "ggml-org/models:F16" -> <"ggml-org/models", "F16">
// tag is optional and can be empty
std::pair<std::string, std::string> common_download_split_repo_tag(const std::string & hf_repo_with_tag);

// download single file from url to local path
// returns status code or -1 on error
// skip_etag: if true, don't read/write .etag files (for HF cache where filename is the hash)
int common_download_file_single(const std::string & url,
                                const std::string & path,
                                const std::string & bearer_token = "",
                                bool skip_etag = false);

// download the given <url, local_path> files in parallel, throws on failure
void common_download_files(const std::vector<std::pair<std::string, std::string>> & url_paths,
                           const std::string & bearer_token = "",
                           bool skip_etag = true);

struct common_hf_download_result {
    std::string model_path;  // local path of the model (first split, if sharded)
    std::string mmproj_path; // local path of the mmproj file, empty if none found
};

// resolve and download the model (and optionally the mmproj sibling) from a HF
// repo into the HuggingFace hub cache, reusing cached files when available
// hf_repo format: <user>/<model>[:quant]
// hf_file: exact file within the repo (overrides the :quant tag, may be empty)
// hf_token: HF access token, may be empty
// throws std::runtime_error on failure
common_hf_download_result common_download_hf_model(
        const std::string & hf_repo,
        const std::string & hf_file,
        const std::string & hf_token,
        bool download_mmproj);
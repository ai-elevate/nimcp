{
  "targets": [
    {
      "target_name": "nimcp_nodejs",
      "sources": [ "binding.c" ],
      "include_dirs": [
        "../../../include"
      ],
      "libraries": [
        "-L<(module_root_dir)/../../../build/lib",
        "-lnimcp"
      ],
      "cflags": ["-fPIC", "-std=c11"],
      "ldflags": ["-Wl,-rpath,<(module_root_dir)/../../../build/lib"]
    }
  ]
}

{
  "targets": [
    {
      "target_name": "nimcp",
      "sources": [ "binding.c" ],
      "include_dirs": [
        "../../"
      ],
      "libraries": [
        "-L/<(module_root_dir)/../../../bin",
        "-lnimcp"
      ],
      "cflags": ["-fPIC", "-std=c11"],
      "ldflags": ["-Wl,-rpath,/<(module_root_dir)/../../../bin"]
    }
  ]
}

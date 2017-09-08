INSTALL
=======

### To install basic tools

Merlin by default requires installation of some basic tools
e.g., SPTK, WORLD

```sh
bash tools/compile_tools.sh
```

### To install other speech tools

When building a new voice, Merlin requires few other tools in order to build labels:
e.g., speech tools, festival and festvox

```sh
bash tools/compile_other_speech_tools.sh
```

If you want to build state align labels, Merlin requires installation of HTK

```sh
bash tools/compile_htk.sh
```

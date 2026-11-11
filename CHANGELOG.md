# Changelog

Releases are cut by python-semantic-release from the Conventional Commits on
`main`; each entry below is generated, so edit commit messages, not this file.

<!-- version list -->

## v0.2.1 (2026-11-11)

### Documentation

- Move the acknowledgements under the citation
  ([`4843d6d`](https://github.com/lawrenceberry/modax/commit/4843d6dae68f55acc00cfc85f61a176fb7cbcdb0))

- Rewrite the docstrings plainly
  ([`766f10f`](https://github.com/lawrenceberry/modax/commit/766f10f3e42702297b60058c8027ba57268dddcc))

### Performance Improvements

- Compile a plain Auto solve once
  ([`ee43847`](https://github.com/lawrenceberry/modax/commit/ee438470e2a381f610d6862428c64a40bcb279fe))

### Refactoring

- Remove duplicated host-side code
  ([`cc9b74c`](https://github.com/lawrenceberry/modax/commit/cc9b74cf4e54d1c531b67b0167f8287ebc679639))


## v0.2.0 (2026-11-11)

### Build System

- Repair uv.lock broken by a merge
  ([`4f7e42e`](https://github.com/lawrenceberry/modax/commit/4f7e42ea2fe82f631f00899b17cb7f8ac6e04034))

### Chores

- Fix the ty errors and E741 failing lint
  ([`43eb759`](https://github.com/lawrenceberry/modax/commit/43eb7592ecbb8be30fc21b13ba2959e6c5f757c3))

### Continuous Integration

- Run type checking as its own workflow
  ([`461387b`](https://github.com/lawrenceberry/modax/commit/461387b487901c96b07e0e47ce3c2bf9055c212d))

### Documentation

- Document only the exported API
  ([`88f6a75`](https://github.com/lawrenceberry/modax/commit/88f6a75e89feb7448c6620d0db71b8324cfe1a1d))

- Drop the theme's top navbar
  ([`6913019`](https://github.com/lawrenceberry/modax/commit/6913019c7d299b32c3bced6c267116622d6b4616))

- Group the methods under API / Methods
  ([`aece2da`](https://github.com/lawrenceberry/modax/commit/aece2dac30584c37a7bfd87fabe52dd30bc87c7b))

- Migrate the site from mkdocs to Sphinx
  ([`3ce88f1`](https://github.com/lawrenceberry/modax/commit/3ce88f15e638419879f91a9dc32f6b385e858cf3))

### Features

- Add Hierarchy.from_sparsity
  ([`54659cd`](https://github.com/lawrenceberry/modax/commit/54659cd004906e024a2b49783c4691add3a5cae3))

- Give modax.solve an explicit signature
  ([`e459cbc`](https://github.com/lawrenceberry/modax/commit/e459cbc5333393a26708ed1c76afcbfcc742dc4c))

### Refactoring

- Import the public API eagerly
  ([`9e8399c`](https://github.com/lawrenceberry/modax/commit/9e8399ca20629ff2fca39fe79f675b975758a7ac))

### Testing

- Skip the hierarchy tests without a GPU
  ([`561888c`](https://github.com/lawrenceberry/modax/commit/561888c6a0050114b37cddcf930ca2376d79c092))


## v0.1.0 (2026-11-11)

### Bug Fixes

- Cap inherited error in controller history
  ([`13afba7`](https://github.com/lawrenceberry/modax/commit/13afba7ad095928d08ff43d0d1fab7d9ed28cf7e))

- Count rejected attempts in Auto's switch
  ([`612ec57`](https://github.com/lawrenceberry/modax/commit/612ec5729efc25661c31b239b4e3162dbcc79edb))

- Damp Tsit5 blocks' inherited error by rates
  ([`a499d4d`](https://github.com/lawrenceberry/modax/commit/a499d4d6ff687a4381de3e3aa5379e14d9ee5ef5))

### Continuous Integration

- Enforce conventional commit messages
  ([`4b7bd95`](https://github.com/lawrenceberry/modax/commit/4b7bd953de70e9513256c061ac67b36f36f719d7))

- Release with python-semantic-release
  ([`b9eb439`](https://github.com/lawrenceberry/modax/commit/b9eb4397fac7f0690c57df9f964444d5e069abe3))

### Features

- Generalise the split to a block hierarchy
  ([`dd254c2`](https://github.com/lawrenceberry/modax/commit/dd254c2f395600ac1ef4ae2962ef9eaf64fb3f81))

- Give each hierarchy block its own method
  ([`307191f`](https://github.com/lawrenceberry/modax/commit/307191f1b9276c101b89147a9b615bcb6bba6ee3))

### Performance Improvements

- Compile each device callee once, not twice
  ([`1dab4e4`](https://github.com/lawrenceberry/modax/commit/1dab4e40a097f0e7e0df00289fc40bc5f6253eeb))

- Seed inherited error as one direction
  ([`3cbca8b`](https://github.com/lawrenceberry/modax/commit/3cbca8b4050f426af8a3ff8f01d0c16872254f15))

- Speed up the hierarchy kernel's step work
  ([`e0e8f95`](https://github.com/lawrenceberry/modax/commit/e0e8f95a972c3a37e5a31de79605f7174dc8bceb))

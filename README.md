<div align="center">

[![npm latest package][npm-latest-image]][npm-url]
[![Build Status][ci-image]][ci-url]
[![License][license-image]][license-url]
[![npm downloads][npm-downloads-image]][npm-url]
[![Follow on Twitter][twitter-image]][twitter-url]

</div>

## Inverter / NPM package Template

Bun + Npm + Typescript + Standard Version + Flat Config Linting + Husky + Commit / Release Pipeline

## Summary

This package contains < a template for devoloping for npm packages > for [<brand_name>](https://github.com/mguleryuz).

Check out the [Changelog](./CHANGELOG.md) to see what changed in the last releases.

## Install

```bash
bun add mongo-rag
```

Install Bun ( bun is the default package manager for this project ( its optional ) ):

```bash
# Supported on macOS, Linux, and WSL
curl -fsSL https://bun.sh/install | bash
# Upgrade Bun every once in a while
bun upgrage
```

## Developing

Install Dependencies:

```bash
bun i
```

Watching TS Problems:

```bash
bun watch
```

## How to make a release

**For the Maintainer**: Add `NPM_TOKEN` to the GitHub Secrets.

1. PR with changes
2. Merge PR into main
3. Checkout main
4. `git pull`
5. `bun release: '' | alpha | beta` optionally add `-- --release-as minor | major | 0.0.1`
6. Make sure everything looks good (e.g. in CHANGELOG.md)
7. Lastly run `bun release:pub`
8. Done

## License

This package is licensed - see the [LICENSE](./LICENSE) file for details.

[ci-image]: https://badgen.net/github/checks/mguleryuz/mongo-rag/main?label=ci
[ci-url]: https://github.com/mguleryuz/mongo-rag/actions/workflows/ci.yaml
[npm-url]: https://npmjs.org/package/mongo-rag
[twitter-url]: https://twitter.com/mgguleryuz
[twitter-image]: https://img.shields.io/twitter/follow/mgguleryuz.svg?label=follow+Mguleryuz
[license-image]: https://img.shields.io/badge/License-LGPL%20v3-blue
[license-url]: ./LICENSE
[npm-latest-image]: https://img.shields.io/npm/v/mongo-rag/latest.svg
[npm-downloads-image]: https://img.shields.io/npm/dm/mongo-rag.svg

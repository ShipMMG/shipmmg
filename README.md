# ShipMMG: Ship Maneuvering Simulation Model

[![PyPI version](https://badge.fury.io/py/shipmmg.svg)](https://badge.fury.io/py/shipmmg)
[![Anaconda-Server Badge](https://anaconda.org/taiga4112/shipmmg/badges/version.svg)](https://anaconda.org/taiga4112/shipmmg)
![codecov](https://github.com/ShipMMG/shipmmg/workflows/codecov/badge.svg)
[![codecov](https://codecov.io/gh/ShipMMG/shipmmg/branch/main/graph/badge.svg?token=VQ1J2RTC7X)](https://codecov.io/gh/ShipMMG/shipmmg)

## What is it?

**ShipMMG** is a unofficial Python package of ship maneuvering simulation with respect to the research committee on “standardization of mathematical model for ship maneuvering predictions” was organized by the JASNAOE.

## Where to get it

The source code is currently hosted on GitHub at: [https://github.com/ShipMMG/shipmmg](https://github.com/ShipMMG/shipmmg)

Binary installers for the latest released version will be available at the Python package index. Now, please install pDESy as following.

```sh
pip install shipmmg
# pip install git+ssh://git@github.com/ShipMMG/shipmmg.git # Install from GitHub
# conda install -c conda-forge -c taiga4112 shipmmg # Install from Anaconda
```

## License

[MIT](https://github.com/ShipMMG/shipmmg/blob/master/LICENSE)

## For developers

### Developing shipmmg API

Here is an example of constructing a developing environment.

```sh
docker build -t shipmmg-dev-env .
docker run --rm --name shipmmg-dev -v `pwd`:/code -w /code -it shipmmg-dev-env /bin/bash
```

In this docker container, we can run `pytest` for checking this library.

### Checking shipmmg API

Here is an example of checking the shipmmg developing version using JupyterLab.

```sh
docker-compose build
docker-compose up
```

After that, access [http://localhost:8888](http://localhost:8888).

- Password is `shipmmg`.

## Contribution

1. Fork it ( <http://github.com/ShipMMG/shipmmg/fork> )
2. Create your feature branch (git checkout -b my-new-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin my-new-feature)
5. Create new Pull Request

If you want to join this project as a researcher, please contact [me](https://github.com/taiga4112).

# DDM Detection Dashboard and Notebooks

A simple dashboard 

## Installation

It's recommended to have [uv](https://docs.astral.sh/uv/).

After cloning the repo, open your terminal and go to the directory and create an environment:

```bash
uv venv --python 3.12
```

Then activate it and sync the repo:

```bash
source .venv/bin/activate

uv sync
```


## Quickstart

You need to [download the data](https://drive.google.com/file/d/1z3RcNre72ELwEa_GO0qsXSh5V2qzdy3q/view?usp=sharing) and put it in the `data` directory for both the app and the notebooks to work

### The dashboard app

After activating the environment you can run the bokeh app as follows:

```bash
uv run bokeh serve --show --app.py
```

### The notebooks

You can run the notebooks normally after the data is downloaded. The important part is making sure the data is in the correct place:

```python


DATA_ROOT = Path(sys.executable).parents[2] / "data"
DATA_ROOT

>>> PosixPath('<your-path-to-repo>/ddm_dashboard/data')
```

# Setting up the Conda Environment

You're in the `buy-vs-invest` directory. Follow these steps:

## 1. Create the conda environment from the environment.yml file

```bash
conda env create -f environment.yml
```

This will:
- Create a new conda environment named `buy-vs-invest`
- Install Python 3.12
- Install numpy, pandas, streamlit, plotly, and all dependencies
- Take ~2-3 minutes (first time only)

## 2. Activate the environment

```bash
conda activate buy-vs-invest
```

You should see `(buy-vs-invest)` appear at the start of your terminal prompt.

## 3. Verify the installation

```bash
python --version
pip list | grep -E "streamlit|plotly|pandas|numpy"
```

## 4. Run the app

```bash
streamlit run app.py
```

This will open the app in your browser at http://localhost:8501.

## 5. (Optional) Run the tests

```bash
python test_model.py
```

## Useful conda commands

- **See all your environments**: `conda env list`
- **Deactivate**: `conda deactivate`
- **Remove the environment**: `conda env remove --name buy-vs-invest`
- **Update packages**: `conda env update -f environment.yml --prune`

## Troubleshooting

If you get permission errors on macOS/Linux:
- Make sure you're in the `buy-vs-invest` directory
- Try `conda env create --file environment.yml` instead

If Streamlit doesn't find dependencies:
- Double-check: `conda activate buy-vs-invest` (you should see the prefix in your prompt)
- Run `which python` — should point to your conda env, not system Python

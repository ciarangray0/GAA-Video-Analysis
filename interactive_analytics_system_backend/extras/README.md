# Extras

Optional components that are not part of the core API deployment.

## streamlit_app.py

An alternative Streamlit-based frontend for local development and testing.

### Usage

```bash
# Install Streamlit
pip install streamlit requests

# Run the app (requires the API backend to be running)
streamlit run extras/streamlit_app.py
```

The Streamlit app expects the API to be running at `http://localhost:8000`.


"""A tiny stand-in for streamlit so app.py can be smoke-tested without the package (CI / sandbox)."""
import types


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return getattr(_st, name)


def _noop(*a, **k):
    return None


def _first(label, options=(), *a, **k):
    options = list(options)
    return options[0] if options else None


_st = types.ModuleType("streamlit")
for n in ["set_page_config", "title", "caption", "header", "subheader", "markdown", "write", "info", "warning", "success", "error",
          "dataframe", "table", "image", "code", "graphviz_chart", "metric", "download_button", "json", "spinner_noop", "rerun"]:
    setattr(_st, n, _noop)
_st.selectbox = _first
_st.multiselect = lambda label, options=(), default=None, **k: list(default if default is not None else [])
_st.button = lambda *a, **k: False
_st.text_input = lambda label, value="", **k: value
_st.slider = lambda label, a=0.0, b=1.0, v=0.5, *x, **k: v
_st.file_uploader = lambda *a, **k: None
_st.data_editor = lambda df, **k: df
_st.tabs = lambda names: [_Ctx() for _ in names]
_st.columns = lambda spec, **k: [_Ctx() for _ in range(spec if isinstance(spec, int) else len(spec))]
_st.spinner = lambda *a, **k: _Ctx()
_st.cache_resource = lambda f=None, **k: f if f else (lambda g: g)
_st.cache_data = _st.cache_resource
_st.sidebar = _Ctx()
_st.chat_input = lambda *a, **k: None
_st.chat_message = lambda *a, **k: _Ctx()
_st.expander = lambda *a, **k: _Ctx()
_st.toggle = lambda label, value=False, **k: value
_st.secrets = {}
_st.session_state = {}

from jinja2 import Environment, FileSystemLoader
from report_agent.dbt_context.from_docs_json import load_manifest, get_model_node, get_column_metadata
from report_agent.utils.config_loader import load_configs

env = Environment(
    loader=FileSystemLoader(searchpath="report_agent/nlg/templates"),
    autoescape=True,
)

def build_prompt(model: str, df_sample, delta, anomalies, alerts) -> str:
    """
    Renders a Jinja template into the LLM prompt.
    Passes in:
      • model_name
      • model_description
      • column_metadata
      • df_sample (as list of dicts)
      • delta, anomalies, alerts dicts
    """
    cfg      = load_configs()
    manifest = load_manifest(cfg)
    node     = get_model_node(manifest, model)
    col_meta = get_column_metadata(node)

    template = env.get_template("weekly_report_prompt.j2")
    return template.render(
        model_name=model,
        model_desc=node["description"],
        columns=col_meta,
        sample=df_sample.to_dict(orient="records"),
        delta=delta,
        anomalies=anomalies,
        alerts=alerts,
    )
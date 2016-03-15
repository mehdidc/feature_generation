from flask import Flask, render_template
from importlib import import_module
from lightexperiments.light import Light

app = Flask(__name__, static_folder='annotate_gen')
app.debug = True


@app.route('/')
def index():
    return render_template("index.html")

if __name__ == '__main__':
    import argparse
    import json
    light = Light()
    light.launch()
    db = light.db
    assert db is not None

    parser = argparse.ArgumentParser(description='WebServer')
    parser.add_argument("--host",
                        type=str,
                        default="0.0.0.0")
    parser.add_argument("--port",
                        type=int,
                        default=5000)
    parser.add_argument("--name", type=str, default="annotate")
    parser.add_argument("--options", type=str, default="{}")
    args = parser.parse_args()
    module = import_module(args.name)
    options = json.loads(args.options)
    view = module.build(db=db, **options)
    app.add_url_rule('/sample', view_func=view.as_view("sample"))
    app.run(host=args.host, port=args.port)

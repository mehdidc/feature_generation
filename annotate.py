from flask import render_template
from flask.views import View
from flask_wtf import Form
from wtforms import SubmitField, SelectMultipleField, SelectField, HiddenField
from tasks import check
from skimage.io import imsave
import uuid
import json
import os


def build(db, expid=None, model_filename="model_E.pkl", dataset="digits"):
    if expid is None:
        expid = str(uuid.uuid4())
    folder = "annotate_gen/{}".format(expid)
    if not os.path.exists(folder):
        os.mkdir(folder)

    nb = 100
    perform = check(filename=model_filename,
                    what="genetic",
                    dataset=dataset,
                    opname='random',
                    params=json.dumps({"k": 2, "nb": nb}),
                    just_get_function=True)
    cats = (('0', 'bad'), ('0', 'nice'))
    desc = ""

    state = {"index": 0, "cur_samples": None}

    def url_sampler():
        if state["index"] % nb == 0:
            state["cur_samples"] = perform()
            state["index"] = 0
        name = str(uuid.uuid4()) + ".png"
        filename = "{}/{}".format(folder, name)
        y = state["cur_samples"][state["index"]]
        state["index"] += 1
        print(y.shape)
        imsave(filename, y)
        url = filename
        return {"url": url}

    class Choice(Form):
        choice = SelectField(choices=cats)
        submit = SubmitField()
        url = HiddenField()

    class Main(View):
        methods = ('POST', 'GET')

        def dispatch_request(self):
            form = Choice(csrf_enabled=False)
            if form.validate_on_submit():
                output = {"choice": form.choice.data,
                          "url": form.url.data,
                          "expid": expid,
                          "tags": ["annotate"]}
                db.insert(output)
            sampled = url_sampler()
            return render_template("imagelabeling.html",
                                   form=form,
                                   desc=desc,
                                   **sampled)

    return Main

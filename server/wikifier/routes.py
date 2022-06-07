from flask import Blueprint
from flask import render_template
from flask import request

#from server.core import long_sent_finder

bp = Blueprint('wikifier', __name__, url_prefix='/wikifier')


@bp.route('/', methods=['POST', 'GET'])
def index():
    """
    Index page for the wikifier app
    route: `/wikifier/`
    This page shows the main wikifier application.
    """
    if request.method == 'POST':
        text = request.form['text'].strip()
        return render_template(
            'wikifier/index.html',
            form_data={
                'text': text,
            },
        )

    return render_template(
        'wikifier/index.html',
    )


@bp.route('/wikify', methods=['POST'])
def wikify():

    text = request.form['text']

    if len(text) == 0:
        return render_template(
            'wikifier/wikify.html',
            error='Voer een geldige tekst in!',
        )

    return render_template(
        'wikifier/index.html',
        text=text,
    )

from flask import Blueprint
from flask import render_template
from flask import request

from nltk import word_tokenize
from nltk import pos_tag

from server.core.wikify import Token
from server.core.wikify import wikify as core_wikify

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

    word_tokens = word_tokenize(text)
    pos_tags = pos_tag(word_tokens)

    tokens = []
    for token, pos in zip(word_tokens, pos_tags):
        nt = Token(
            start_off=0,
            end_off=0,
            id_=0,
            token=token,
            pos=pos[1],
            entity=None,
            link=None,
            core_nlp_ent=None,
        )
        tokens.append(nt)

    wikified_tokens = core_wikify(tokens, url='http://localhost:8126')

    # TODO: Add inline links for all tokens that have links!

    return render_template(
        'wikifier/wikify.html',
        text=text,
        tokens=wikified_tokens,
    )

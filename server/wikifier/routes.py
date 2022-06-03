from flask import Blueprint
from flask import render_template
from flask import request

bp = Blueprint('wikifier', __name__, url_prefix='/wikifier')

@bp.route('/')
def index():
    """
    Index page for the assistant app
    route: `/wikifier/`
    This page shows the main assistant application.
    """

    return render_template(
        'assistant/index.html',
    )

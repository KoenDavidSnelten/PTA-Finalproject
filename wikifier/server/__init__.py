import os

from flask import Flask
from flask import render_template


def create_app():
    """
    Create and configure an instance of the Flask application.
    """

    app = Flask(__name__)

    # Makes sure that all application folders are created.
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/')
    def index():
        """
        Index page for the website.
        route: `/`
        This page shows some basis information about the
        website and allows users to navigate to their desired pages.
        """
        return render_template('index.html')

    from wikifier.server.wikifier.routes import bp as assistant_bp
    app.register_blueprint(assistant_bp)

    return app

from flask import Flask
from config import FLASK_SECRET_KEY

def create_app():
    app = Flask(__name__,
                template_folder='templates',
                static_folder='../static')
    app.secret_key = FLASK_SECRET_KEY

    from app.routes import bp
    app.register_blueprint(bp)
    return app